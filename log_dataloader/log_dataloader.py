
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fc4b1425ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fc4b1425ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fc51c095510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fc51c095510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fc53b2c2ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fc53b2c2ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc4c943da60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc4c943da60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc4c943da60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 32%|███▏      | 3186688/9912422 [00:00<00:00, 31360056.28it/s]9920512it [00:00, 30381493.75it/s]                             
0it [00:00, ?it/s]32768it [00:00, 561745.71it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155128.93it/s]1654784it [00:00, 10986953.30it/s]                         
0it [00:00, ?it/s]8192it [00:00, 169843.79it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc4b12e06d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc4b12ef978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fc4c943d6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fc4c943d6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fc4c943d6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:28,  1.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:28,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:28,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:27,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:27,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:26,  1.81 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<01:01,  2.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:01,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:00,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:00,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:59,  2.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:41,  3.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:41,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:41,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:41,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:41,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:40,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:40,  3.58 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  4.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:28,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:28,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:28,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:28,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:28,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:27,  4.97 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:20,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:20,  6.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:19,  6.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:01<00:14,  9.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:14,  9.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:14,  9.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:14,  9.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:13,  9.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:13,  9.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:13,  9.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 12.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:09, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:09, 12.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 19.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 23.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:04, 23.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 23.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 23.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 27.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 27.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 27.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 27.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 27.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:03, 27.27 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 29.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 29.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 29.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 29.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 29.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 29.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:02<00:03, 28.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:02<00:03, 28.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:02<00:03, 28.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:02<00:03, 25.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:02<00:03, 25.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:02<00:03, 25.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:02<00:03, 25.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:02<00:03, 25.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:02<00:03, 22.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:02<00:03, 22.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:02<00:03, 22.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:02<00:03, 22.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:02<00:04, 20.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:02<00:04, 20.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:02<00:03, 20.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:02<00:03, 20.85 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:02<00:04, 19.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:02<00:04, 19.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:02<00:04, 19.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:02<00:03, 19.82 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:04, 18.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:02<00:04, 18.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:02<00:04, 18.77 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:03<00:04, 18.77 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:04, 17.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:03<00:04, 17.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:03<00:04, 17.84 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:04, 17.04 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:03<00:04, 17.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:03<00:04, 17.04 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:04, 16.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:03<00:04, 16.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:03<00:04, 16.56 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:04, 15.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:03<00:04, 15.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:03<00:04, 15.96 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:04, 15.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:03<00:04, 15.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:03<00:04, 15.20 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:04, 14.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:03<00:04, 14.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:03<00:04, 14.48 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:03<00:04, 14.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:03<00:04, 14.54 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:04<00:04, 14.54 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:04<00:04, 14.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:04<00:04, 14.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:04<00:04, 14.68 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:03, 14.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:04<00:03, 14.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:04<00:03, 14.84 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:04<00:03, 15.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:04<00:03, 15.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:04<00:03, 15.01 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:04<00:03, 14.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:04<00:03, 14.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:04<00:03, 14.90 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:04<00:03, 15.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:04<00:03, 15.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:04<00:03, 15.12 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:04<00:03, 15.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:04<00:03, 15.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:04<00:03, 15.18 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:04<00:03, 15.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:04<00:03, 15.17 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:04<00:03, 15.17 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:04<00:03, 15.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:04<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:04<00:03, 15.17 MiB/s][A

Extraction completed...: 0 file [00:04, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:05<00:03, 15.17 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:05<00:03, 14.97 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:05<00:03, 14.97 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:05<00:02, 14.97 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:05<00:02, 14.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:05<00:02, 14.36 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:05<00:02, 14.36 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:05<00:02, 14.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:05<00:02, 14.03 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:05<00:02, 14.03 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:05<00:02, 13.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:05<00:02, 13.88 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:05<00:02, 13.88 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:05<00:02, 13.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:05<00:02, 13.64 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:05<00:02, 13.64 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:05<00:02, 13.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:05<00:02, 13.53 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:05<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:05<00:02, 13.53 MiB/s][A

Extraction completed...: 0 file [00:05, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:06<00:02, 13.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:06<00:02, 13.50 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:06<00:02, 13.50 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:06<00:02, 13.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:06<00:02, 13.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:06<00:02, 13.46 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  82%|████████▏ | 133/162 [00:06<00:02, 13.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:06<00:02, 13.36 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:06<00:02, 13.36 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:06<00:02, 13.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:06<00:02, 13.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:06<00:01, 13.42 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:06<00:01, 13.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:06<00:01, 13.37 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:06<00:01, 13.37 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:06<00:01, 13.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:06<00:01, 13.30 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:06<00:01, 13.30 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:06<00:01, 13.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:06<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:06<00:01, 13.34 MiB/s][A

Extraction completed...: 0 file [00:06, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:07<00:01, 13.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:07<00:01, 13.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:07<00:01, 13.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:07<00:01, 13.34 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:07<00:01, 13.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:07<00:01, 13.25 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:07<00:01, 13.25 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:07<00:01, 13.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:07<00:01, 13.20 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:07<00:01, 13.20 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:07<00:01, 12.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:07<00:01, 12.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:07<00:00, 12.83 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:00, 12.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:07<00:00, 12.40 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:07<00:00, 12.40 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:07<00:00, 11.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:07<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:07<00:00, 11.95 MiB/s][A

Extraction completed...: 0 file [00:07, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:08<00:00, 11.95 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:08<00:00, 11.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:08<00:00, 11.52 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:08<00:00, 11.52 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:08<00:00, 10.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:08<00:00, 10.90 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:08<00:00, 10.90 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:08<00:00, 10.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:08<00:00, 10.49 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:08<00:00, 10.49 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:08<00:00, 10.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:08<00:00, 10.27 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:08<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 10.27 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.84s/ url]Dl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 10.27 MiB/s][A

Extraction completed...: 0 file [00:08, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:08<00:00,  8.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:08<00:00, 10.27 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:08<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.68s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:10<00:00,  8.84s/ url]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00, 10.27 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.68s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:10<00:00, 10.68s/ file]
Dl Size...: 100%|██████████| 162/162 [00:10<00:00, 15.17 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:10<00:00, 10.68s/ url]
0 examples [00:00, ? examples/s]2020-07-27 00:08:35.810339: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-27 00:08:35.822993: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-27 00:08:35.823160: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5596dd887190 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-27 00:08:35.823176: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
34 examples [00:00, 338.41 examples/s]148 examples [00:00, 428.59 examples/s]263 examples [00:00, 527.39 examples/s]385 examples [00:00, 635.25 examples/s]502 examples [00:00, 734.98 examples/s]622 examples [00:00, 830.59 examples/s]748 examples [00:00, 924.45 examples/s]872 examples [00:00, 999.25 examples/s]999 examples [00:00, 1067.35 examples/s]1124 examples [00:01, 1114.20 examples/s]1244 examples [00:01, 1138.45 examples/s]1365 examples [00:01, 1157.80 examples/s]1493 examples [00:01, 1189.95 examples/s]1616 examples [00:01, 1201.27 examples/s]1744 examples [00:01, 1221.85 examples/s]1868 examples [00:01, 1217.98 examples/s]1991 examples [00:01, 1197.15 examples/s]2114 examples [00:01, 1206.39 examples/s]2236 examples [00:01, 1190.38 examples/s]2356 examples [00:02, 1166.51 examples/s]2476 examples [00:02, 1175.05 examples/s]2599 examples [00:02, 1189.77 examples/s]2719 examples [00:02, 1191.61 examples/s]2841 examples [00:02, 1197.85 examples/s]2966 examples [00:02, 1211.40 examples/s]3088 examples [00:02, 1204.85 examples/s]3211 examples [00:02, 1212.16 examples/s]3335 examples [00:02, 1218.49 examples/s]3459 examples [00:02, 1223.09 examples/s]3583 examples [00:03, 1226.84 examples/s]3706 examples [00:03, 1208.00 examples/s]3832 examples [00:03, 1221.02 examples/s]3956 examples [00:03, 1223.87 examples/s]4083 examples [00:03, 1234.69 examples/s]4211 examples [00:03, 1246.32 examples/s]4336 examples [00:03, 1165.58 examples/s]4462 examples [00:03, 1192.28 examples/s]4587 examples [00:03, 1208.53 examples/s]4709 examples [00:03, 1208.39 examples/s]4831 examples [00:04, 1207.00 examples/s]4954 examples [00:04, 1212.58 examples/s]5078 examples [00:04, 1220.57 examples/s]5201 examples [00:04, 1203.31 examples/s]5324 examples [00:04, 1209.05 examples/s]5446 examples [00:04, 1195.36 examples/s]5566 examples [00:04, 1182.99 examples/s]5686 examples [00:04, 1187.70 examples/s]5808 examples [00:04, 1195.59 examples/s]5929 examples [00:04, 1198.04 examples/s]6053 examples [00:05, 1208.49 examples/s]6177 examples [00:05, 1216.08 examples/s]6301 examples [00:05, 1221.08 examples/s]6424 examples [00:05, 1200.70 examples/s]6547 examples [00:05, 1206.54 examples/s]6668 examples [00:05, 1190.93 examples/s]6788 examples [00:05, 1174.98 examples/s]6908 examples [00:05, 1181.64 examples/s]7027 examples [00:05, 1172.74 examples/s]7145 examples [00:06, 1173.02 examples/s]7263 examples [00:06, 1166.03 examples/s]7380 examples [00:06, 1161.17 examples/s]7498 examples [00:06, 1164.33 examples/s]7615 examples [00:06, 1162.80 examples/s]7732 examples [00:06, 1162.95 examples/s]7849 examples [00:06, 1161.07 examples/s]7966 examples [00:06, 1163.60 examples/s]8084 examples [00:06, 1167.42 examples/s]8201 examples [00:06, 1152.95 examples/s]8317 examples [00:07, 1083.48 examples/s]8427 examples [00:07, 1085.88 examples/s]8537 examples [00:07, 1044.50 examples/s]8644 examples [00:07, 1051.49 examples/s]8750 examples [00:07, 1045.24 examples/s]8870 examples [00:07, 1086.86 examples/s]8991 examples [00:07, 1120.60 examples/s]9106 examples [00:07, 1127.60 examples/s]9229 examples [00:07, 1154.52 examples/s]9355 examples [00:07, 1181.86 examples/s]9474 examples [00:08, 1158.67 examples/s]9595 examples [00:08, 1172.71 examples/s]9718 examples [00:08, 1187.69 examples/s]9843 examples [00:08, 1204.42 examples/s]9965 examples [00:08, 1208.33 examples/s]10087 examples [00:08, 1151.45 examples/s]10205 examples [00:08, 1159.61 examples/s]10327 examples [00:08, 1175.22 examples/s]10447 examples [00:08, 1181.26 examples/s]10571 examples [00:08, 1198.21 examples/s]10695 examples [00:09, 1210.24 examples/s]10817 examples [00:09, 1211.87 examples/s]10942 examples [00:09, 1220.54 examples/s]11065 examples [00:09, 1185.03 examples/s]11193 examples [00:09, 1209.85 examples/s]11315 examples [00:09, 1199.90 examples/s]11436 examples [00:09, 1150.55 examples/s]11552 examples [00:09, 1123.68 examples/s]11670 examples [00:09, 1137.56 examples/s]11792 examples [00:10, 1159.85 examples/s]11915 examples [00:10, 1177.68 examples/s]12036 examples [00:10, 1185.88 examples/s]12159 examples [00:10, 1196.13 examples/s]12279 examples [00:10, 1190.83 examples/s]12401 examples [00:10, 1198.93 examples/s]12523 examples [00:10, 1204.62 examples/s]12644 examples [00:10, 1190.91 examples/s]12764 examples [00:10, 1187.02 examples/s]12885 examples [00:10, 1192.28 examples/s]13009 examples [00:11, 1204.55 examples/s]13130 examples [00:11, 1204.63 examples/s]13251 examples [00:11, 1203.29 examples/s]13375 examples [00:11, 1214.00 examples/s]13497 examples [00:11, 1187.65 examples/s]13619 examples [00:11, 1194.46 examples/s]13742 examples [00:11, 1202.20 examples/s]13863 examples [00:11, 1198.00 examples/s]13985 examples [00:11, 1204.38 examples/s]14106 examples [00:11, 1204.15 examples/s]14227 examples [00:12, 1198.36 examples/s]14348 examples [00:12, 1201.77 examples/s]14469 examples [00:12, 1202.84 examples/s]14591 examples [00:12, 1205.61 examples/s]14715 examples [00:12, 1215.30 examples/s]14838 examples [00:12, 1216.70 examples/s]14963 examples [00:12, 1225.06 examples/s]15086 examples [00:12, 1215.63 examples/s]15210 examples [00:12, 1222.38 examples/s]15334 examples [00:12, 1227.40 examples/s]15457 examples [00:13, 1223.35 examples/s]15580 examples [00:13, 1217.60 examples/s]15702 examples [00:13, 1194.58 examples/s]15822 examples [00:13, 1164.82 examples/s]15942 examples [00:13, 1174.73 examples/s]16060 examples [00:13, 1153.69 examples/s]16176 examples [00:13, 1126.10 examples/s]16289 examples [00:13, 1104.94 examples/s]16413 examples [00:13, 1140.92 examples/s]16534 examples [00:14, 1158.98 examples/s]16656 examples [00:14, 1174.34 examples/s]16783 examples [00:14, 1200.41 examples/s]16904 examples [00:14, 1197.89 examples/s]17025 examples [00:14, 1197.19 examples/s]17149 examples [00:14, 1207.66 examples/s]17274 examples [00:14, 1220.01 examples/s]17401 examples [00:14, 1233.21 examples/s]17525 examples [00:14, 1174.57 examples/s]17644 examples [00:14, 1166.69 examples/s]17762 examples [00:15, 1165.83 examples/s]17879 examples [00:15, 1121.12 examples/s]17992 examples [00:15, 1106.37 examples/s]18105 examples [00:15, 1113.19 examples/s]18225 examples [00:15, 1137.53 examples/s]18346 examples [00:15, 1156.19 examples/s]18469 examples [00:15, 1176.32 examples/s]18587 examples [00:15, 1160.06 examples/s]18704 examples [00:15, 1162.97 examples/s]18821 examples [00:15, 1151.96 examples/s]18937 examples [00:16, 1150.69 examples/s]19053 examples [00:16, 1146.54 examples/s]19168 examples [00:16, 1134.34 examples/s]19287 examples [00:16, 1148.88 examples/s]19407 examples [00:16, 1163.60 examples/s]19524 examples [00:16, 1136.64 examples/s]19638 examples [00:16, 1136.49 examples/s]19757 examples [00:16, 1151.68 examples/s]19873 examples [00:16, 1122.51 examples/s]19993 examples [00:16, 1142.12 examples/s]20108 examples [00:17, 1094.07 examples/s]20226 examples [00:17, 1117.24 examples/s]20339 examples [00:17, 1087.41 examples/s]20449 examples [00:17, 1085.47 examples/s]20560 examples [00:17, 1090.84 examples/s]20671 examples [00:17, 1096.33 examples/s]20781 examples [00:17, 1096.83 examples/s]20896 examples [00:17, 1110.43 examples/s]21019 examples [00:17, 1141.30 examples/s]21142 examples [00:18, 1164.42 examples/s]21265 examples [00:18, 1181.02 examples/s]21387 examples [00:18, 1192.09 examples/s]21511 examples [00:18, 1204.69 examples/s]21632 examples [00:18, 1203.12 examples/s]21754 examples [00:18, 1207.96 examples/s]21876 examples [00:18, 1211.07 examples/s]21998 examples [00:18, 1211.24 examples/s]22120 examples [00:18, 1209.53 examples/s]22243 examples [00:18, 1213.71 examples/s]22366 examples [00:19, 1218.37 examples/s]22491 examples [00:19, 1226.91 examples/s]22614 examples [00:19, 1205.77 examples/s]22735 examples [00:19, 1181.49 examples/s]22854 examples [00:19, 1182.22 examples/s]22979 examples [00:19, 1199.91 examples/s]23102 examples [00:19, 1208.58 examples/s]23223 examples [00:19, 1207.13 examples/s]23344 examples [00:19, 1172.88 examples/s]23462 examples [00:19, 1172.13 examples/s]23581 examples [00:20, 1175.60 examples/s]23699 examples [00:20, 1165.71 examples/s]23822 examples [00:20, 1182.03 examples/s]23941 examples [00:20, 1182.94 examples/s]24063 examples [00:20, 1191.31 examples/s]24184 examples [00:20, 1195.41 examples/s]24304 examples [00:20, 1180.53 examples/s]24423 examples [00:20, 1181.87 examples/s]24544 examples [00:20, 1189.93 examples/s]24669 examples [00:20, 1205.70 examples/s]24792 examples [00:21, 1211.94 examples/s]24914 examples [00:21, 1198.98 examples/s]25034 examples [00:21, 1182.97 examples/s]25153 examples [00:21, 1176.36 examples/s]25271 examples [00:21, 1148.26 examples/s]25387 examples [00:21, 1145.92 examples/s]25502 examples [00:21, 1143.71 examples/s]25617 examples [00:21, 1097.81 examples/s]25732 examples [00:21, 1111.11 examples/s]25844 examples [00:22, 1108.11 examples/s]25956 examples [00:22, 1109.37 examples/s]26078 examples [00:22, 1139.00 examples/s]26198 examples [00:22, 1155.95 examples/s]26316 examples [00:22, 1161.71 examples/s]26439 examples [00:22, 1178.78 examples/s]26560 examples [00:22, 1186.94 examples/s]26683 examples [00:22, 1197.87 examples/s]26804 examples [00:22, 1198.71 examples/s]26924 examples [00:22, 1196.26 examples/s]27046 examples [00:23, 1200.92 examples/s]27169 examples [00:23, 1208.00 examples/s]27293 examples [00:23, 1214.57 examples/s]27419 examples [00:23, 1226.25 examples/s]27544 examples [00:23, 1231.79 examples/s]27668 examples [00:23, 1196.44 examples/s]27788 examples [00:23, 1181.50 examples/s]27911 examples [00:23, 1193.86 examples/s]28034 examples [00:23, 1202.58 examples/s]28155 examples [00:23, 1194.72 examples/s]28277 examples [00:24, 1200.44 examples/s]28399 examples [00:24, 1205.79 examples/s]28524 examples [00:24, 1215.85 examples/s]28646 examples [00:24, 1211.85 examples/s]28768 examples [00:24, 1195.28 examples/s]28888 examples [00:24, 1196.03 examples/s]29010 examples [00:24, 1202.43 examples/s]29131 examples [00:24, 1199.13 examples/s]29251 examples [00:24, 1196.13 examples/s]29371 examples [00:24, 1168.27 examples/s]29491 examples [00:25, 1176.09 examples/s]29614 examples [00:25, 1188.96 examples/s]29734 examples [00:25, 1188.40 examples/s]29853 examples [00:25, 1184.12 examples/s]29973 examples [00:25, 1186.52 examples/s]30092 examples [00:25, 1131.32 examples/s]30217 examples [00:25, 1162.60 examples/s]30334 examples [00:25, 1159.11 examples/s]30451 examples [00:25, 1061.16 examples/s]30560 examples [00:26, 1043.83 examples/s]30666 examples [00:26, 1003.92 examples/s]30768 examples [00:26, 1002.45 examples/s]30870 examples [00:26, 983.07 examples/s] 30970 examples [00:26, 925.59 examples/s]31067 examples [00:26, 936.72 examples/s]31162 examples [00:26, 931.72 examples/s]31261 examples [00:26, 946.10 examples/s]31375 examples [00:26, 995.56 examples/s]31493 examples [00:26, 1043.52 examples/s]31614 examples [00:27, 1086.18 examples/s]31739 examples [00:27, 1128.82 examples/s]31863 examples [00:27, 1157.76 examples/s]31983 examples [00:27, 1168.47 examples/s]32101 examples [00:27, 1157.50 examples/s]32219 examples [00:27, 1161.54 examples/s]32344 examples [00:27, 1185.09 examples/s]32466 examples [00:27, 1192.75 examples/s]32586 examples [00:27, 1185.98 examples/s]32705 examples [00:27, 1179.30 examples/s]32824 examples [00:28, 1169.21 examples/s]32947 examples [00:28, 1186.37 examples/s]33068 examples [00:28, 1193.04 examples/s]33188 examples [00:28, 1185.82 examples/s]33307 examples [00:28, 1183.37 examples/s]33426 examples [00:28, 1170.77 examples/s]33544 examples [00:28, 1150.71 examples/s]33668 examples [00:28, 1174.79 examples/s]33788 examples [00:28, 1181.04 examples/s]33910 examples [00:28, 1190.82 examples/s]34030 examples [00:29, 1185.49 examples/s]34149 examples [00:29, 1184.15 examples/s]34269 examples [00:29, 1187.44 examples/s]34390 examples [00:29, 1192.20 examples/s]34510 examples [00:29, 1175.34 examples/s]34628 examples [00:29, 1164.28 examples/s]34751 examples [00:29, 1181.68 examples/s]34872 examples [00:29, 1188.80 examples/s]34996 examples [00:29, 1202.74 examples/s]35117 examples [00:30, 1193.41 examples/s]35237 examples [00:30, 1163.25 examples/s]35361 examples [00:30, 1185.10 examples/s]35481 examples [00:30, 1185.71 examples/s]35601 examples [00:30, 1188.10 examples/s]35721 examples [00:30, 1191.34 examples/s]35841 examples [00:30, 1167.43 examples/s]35958 examples [00:30, 1158.90 examples/s]36075 examples [00:30, 1149.87 examples/s]36193 examples [00:30, 1158.55 examples/s]36310 examples [00:31, 1161.18 examples/s]36431 examples [00:31, 1174.02 examples/s]36549 examples [00:31, 1172.36 examples/s]36669 examples [00:31, 1177.41 examples/s]36790 examples [00:31, 1186.58 examples/s]36909 examples [00:31, 1159.93 examples/s]37026 examples [00:31, 1154.40 examples/s]37144 examples [00:31, 1161.61 examples/s]37261 examples [00:31, 1150.47 examples/s]37377 examples [00:31, 1137.74 examples/s]37495 examples [00:32, 1146.52 examples/s]37610 examples [00:32, 1134.89 examples/s]37728 examples [00:32, 1146.24 examples/s]37850 examples [00:32, 1165.48 examples/s]37967 examples [00:32, 1158.81 examples/s]38083 examples [00:32, 1157.90 examples/s]38204 examples [00:32, 1170.90 examples/s]38324 examples [00:32, 1178.24 examples/s]38447 examples [00:32, 1191.47 examples/s]38567 examples [00:32, 1188.49 examples/s]38686 examples [00:33, 1144.69 examples/s]38801 examples [00:33, 1105.30 examples/s]38917 examples [00:33, 1120.04 examples/s]39034 examples [00:33, 1132.14 examples/s]39149 examples [00:33, 1136.66 examples/s]39263 examples [00:33, 1120.31 examples/s]39382 examples [00:33, 1138.19 examples/s]39497 examples [00:33, 1092.27 examples/s]39610 examples [00:33, 1102.13 examples/s]39721 examples [00:34, 1090.60 examples/s]39831 examples [00:34, 1057.11 examples/s]39941 examples [00:34, 1068.10 examples/s]40049 examples [00:34, 1021.63 examples/s]40155 examples [00:34, 1031.71 examples/s]40267 examples [00:34, 1056.28 examples/s]40374 examples [00:34, 1041.29 examples/s]40485 examples [00:34, 1059.84 examples/s]40592 examples [00:34, 1058.98 examples/s]40701 examples [00:34, 1065.45 examples/s]40813 examples [00:35, 1080.86 examples/s]40925 examples [00:35, 1090.93 examples/s]41044 examples [00:35, 1118.22 examples/s]41161 examples [00:35, 1132.85 examples/s]41281 examples [00:35, 1151.32 examples/s]41403 examples [00:35, 1169.29 examples/s]41521 examples [00:35, 1168.51 examples/s]41639 examples [00:35, 1170.60 examples/s]41757 examples [00:35, 1142.43 examples/s]41872 examples [00:35, 1119.68 examples/s]41985 examples [00:36, 1101.78 examples/s]42097 examples [00:36, 1104.77 examples/s]42218 examples [00:36, 1133.15 examples/s]42335 examples [00:36, 1142.04 examples/s]42450 examples [00:36, 1131.44 examples/s]42564 examples [00:36, 1103.69 examples/s]42675 examples [00:36, 1059.71 examples/s]42785 examples [00:36, 1070.78 examples/s]42893 examples [00:36, 1069.86 examples/s]43012 examples [00:37, 1101.12 examples/s]43133 examples [00:37, 1130.16 examples/s]43253 examples [00:37, 1147.57 examples/s]43369 examples [00:37, 1114.92 examples/s]43482 examples [00:37, 1117.34 examples/s]43595 examples [00:37, 1097.15 examples/s]43706 examples [00:37, 1091.85 examples/s]43821 examples [00:37, 1105.79 examples/s]43932 examples [00:37, 1086.82 examples/s]44041 examples [00:37, 1071.15 examples/s]44155 examples [00:38, 1089.90 examples/s]44267 examples [00:38, 1097.08 examples/s]44383 examples [00:38, 1113.24 examples/s]44501 examples [00:38, 1132.20 examples/s]44620 examples [00:38, 1148.27 examples/s]44736 examples [00:38, 1149.93 examples/s]44852 examples [00:38, 1127.29 examples/s]44972 examples [00:38, 1145.88 examples/s]45092 examples [00:38, 1157.18 examples/s]45216 examples [00:38, 1179.47 examples/s]45340 examples [00:39, 1194.62 examples/s]45465 examples [00:39, 1208.20 examples/s]45587 examples [00:39, 1210.02 examples/s]45711 examples [00:39, 1217.15 examples/s]45835 examples [00:39, 1223.59 examples/s]45958 examples [00:39, 1201.46 examples/s]46079 examples [00:39, 1203.06 examples/s]46201 examples [00:39, 1206.68 examples/s]46322 examples [00:39, 1162.71 examples/s]46445 examples [00:39, 1179.50 examples/s]46567 examples [00:40, 1188.88 examples/s]46688 examples [00:40, 1194.92 examples/s]46808 examples [00:40, 1188.39 examples/s]46929 examples [00:40, 1192.87 examples/s]47052 examples [00:40, 1203.35 examples/s]47175 examples [00:40, 1210.13 examples/s]47297 examples [00:40, 1210.40 examples/s]47419 examples [00:40, 1195.32 examples/s]47540 examples [00:40, 1197.98 examples/s]47665 examples [00:41, 1211.67 examples/s]47789 examples [00:41, 1218.21 examples/s]47911 examples [00:41, 1217.84 examples/s]48033 examples [00:41, 1206.23 examples/s]48154 examples [00:41, 1200.10 examples/s]48280 examples [00:41, 1216.86 examples/s]48402 examples [00:41, 1202.37 examples/s]48523 examples [00:41, 1202.06 examples/s]48644 examples [00:41, 1186.99 examples/s]48763 examples [00:41, 1168.21 examples/s]48881 examples [00:42, 1170.18 examples/s]48999 examples [00:42, 1149.44 examples/s]49120 examples [00:42, 1166.69 examples/s]49241 examples [00:42, 1177.57 examples/s]49365 examples [00:42, 1193.08 examples/s]49485 examples [00:42, 1165.47 examples/s]49604 examples [00:42, 1170.05 examples/s]49722 examples [00:42, 1162.79 examples/s]49842 examples [00:42, 1171.82 examples/s]49961 examples [00:42, 1174.39 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▋        | 8133/50000 [00:00<00:00, 81326.01 examples/s] 44%|████▍     | 22192/50000 [00:00<00:00, 93098.54 examples/s] 72%|███████▏  | 36070/50000 [00:00<00:00, 103298.47 examples/s]                                                                0 examples [00:00, ? examples/s]96 examples [00:00, 952.69 examples/s]203 examples [00:00, 983.21 examples/s]324 examples [00:00, 1039.99 examples/s]444 examples [00:00, 1081.79 examples/s]563 examples [00:00, 1112.07 examples/s]674 examples [00:00, 1110.51 examples/s]792 examples [00:00, 1129.83 examples/s]909 examples [00:00, 1139.31 examples/s]1023 examples [00:00, 1136.55 examples/s]1139 examples [00:01, 1142.26 examples/s]1256 examples [00:01, 1148.65 examples/s]1371 examples [00:01, 1146.72 examples/s]1492 examples [00:01, 1162.98 examples/s]1609 examples [00:01, 1164.57 examples/s]1728 examples [00:01, 1172.04 examples/s]1845 examples [00:01, 1169.79 examples/s]1962 examples [00:01, 1138.46 examples/s]2086 examples [00:01, 1165.05 examples/s]2205 examples [00:01, 1169.56 examples/s]2324 examples [00:02, 1174.82 examples/s]2447 examples [00:02, 1189.63 examples/s]2567 examples [00:02, 1181.58 examples/s]2690 examples [00:02, 1194.41 examples/s]2810 examples [00:02, 1190.65 examples/s]2932 examples [00:02, 1198.42 examples/s]3052 examples [00:02, 1178.13 examples/s]3170 examples [00:02, 1177.12 examples/s]3288 examples [00:02, 1177.27 examples/s]3406 examples [00:02, 1160.63 examples/s]3525 examples [00:03, 1168.35 examples/s]3644 examples [00:03, 1173.90 examples/s]3767 examples [00:03, 1189.63 examples/s]3887 examples [00:03, 1184.48 examples/s]4006 examples [00:03, 1182.36 examples/s]4125 examples [00:03, 1173.85 examples/s]4243 examples [00:03, 1148.63 examples/s]4360 examples [00:03, 1154.52 examples/s]4476 examples [00:03, 1136.91 examples/s]4593 examples [00:03, 1145.58 examples/s]4720 examples [00:04, 1177.93 examples/s]4845 examples [00:04, 1191.11 examples/s]4967 examples [00:04, 1197.10 examples/s]5087 examples [00:04, 1195.87 examples/s]5207 examples [00:04, 1189.76 examples/s]5332 examples [00:04, 1205.61 examples/s]5457 examples [00:04, 1218.13 examples/s]5582 examples [00:04, 1224.78 examples/s]5707 examples [00:04, 1230.80 examples/s]5831 examples [00:04, 1216.08 examples/s]5953 examples [00:05, 1214.79 examples/s]6075 examples [00:05, 1209.96 examples/s]6197 examples [00:05, 1188.11 examples/s]6316 examples [00:05, 1179.76 examples/s]6435 examples [00:05, 1174.31 examples/s]6553 examples [00:05, 1154.27 examples/s]6678 examples [00:05, 1180.31 examples/s]6804 examples [00:05, 1200.65 examples/s]6929 examples [00:05, 1212.63 examples/s]7051 examples [00:05, 1198.56 examples/s]7174 examples [00:06, 1206.15 examples/s]7298 examples [00:06, 1215.61 examples/s]7423 examples [00:06, 1223.21 examples/s]7548 examples [00:06, 1228.31 examples/s]7671 examples [00:06, 1190.88 examples/s]7791 examples [00:06, 1151.04 examples/s]7912 examples [00:06, 1165.24 examples/s]8036 examples [00:06, 1185.60 examples/s]8157 examples [00:06, 1189.89 examples/s]8277 examples [00:07, 1177.84 examples/s]8402 examples [00:07, 1197.82 examples/s]8523 examples [00:07, 1150.94 examples/s]8639 examples [00:07, 1129.26 examples/s]8755 examples [00:07, 1137.39 examples/s]8870 examples [00:07, 1123.86 examples/s]8988 examples [00:07, 1139.67 examples/s]9103 examples [00:07, 1133.99 examples/s]9219 examples [00:07, 1139.91 examples/s]9334 examples [00:07, 1109.66 examples/s]9451 examples [00:08, 1124.50 examples/s]9566 examples [00:08, 1130.88 examples/s]9684 examples [00:08, 1143.61 examples/s]9799 examples [00:08, 1098.71 examples/s]9910 examples [00:08, 1047.42 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7WO751/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete7WO751/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc4c943da60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc4c943da60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc4c943da60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc514daa0b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc4c74fe240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fc4c943da60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fc4c943da60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fc4c943da60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc4b12ef358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fc4b12ef1d0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fc441d562f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fc441d562f0> 

  function with postional parmater data_info <function split_train_valid at 0x7fc441d562f0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fc441d56400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fc441d56400> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fc441d56400> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.48.0)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.1)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.11/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=8034603d478190da9769de10a60a76a96b7cff8308cf165d2e656a9a9373114e
  Stored in directory: /tmp/pip-ephem-wheel-cache-dixl0ofp/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<22:10:28, 10.8kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:45:07, 15.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<11:04:45, 21.6kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:45:47, 30.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:25:13, 44.0kB/s].vector_cache/glove.6B.zip:   1%|          | 9.27M/862M [00:01<3:46:16, 62.8kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.0M/862M [00:01<2:37:25, 89.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.7M/862M [00:01<1:49:32, 128kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.4M/862M [00:01<1:16:15, 183kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.2M/862M [00:01<53:06, 260kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 37.9M/862M [00:02<37:01, 371kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.9M/862M [00:02<25:50, 528kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.4M/862M [00:02<18:07, 750kB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<12:56, 1.04MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<10:56, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<09:27, 1.42MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:04<06:59, 1.92MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:05<05:04, 2.63MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<16:49, 794kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.6M/862M [00:06<13:12, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:06<09:32, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.4M/862M [00:08<09:37, 1.38MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.6M/862M [00:08<09:24, 1.41MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:08<07:08, 1.86MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:09<05:08, 2.57MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<13:24, 987kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 68.9M/862M [00:10<10:44, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.5M/862M [00:10<07:47, 1.69MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.7M/862M [00:12<08:32, 1.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<08:37, 1.53MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<06:41, 1.96MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<06:47, 1.93MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:30, 1.74MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:14<05:48, 2.25MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:14<04:13, 3.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.9M/862M [00:16<10:18, 1.26MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<08:31, 1.53MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.8M/862M [00:16<06:17, 2.07MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:17<05:06, 2.54MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<9:46:34, 22.1kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<6:50:50, 31.5kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:18<4:46:48, 45.0kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<3:28:15, 61.9kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.1M/862M [00:20<2:28:28, 86.8kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<1:44:25, 123kB/s] .vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:20<1:12:57, 176kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<1:10:00, 183kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<50:17, 255kB/s]  .vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<35:27, 361kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<27:45, 459kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<21:58, 580kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:24<16:00, 795kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<13:13, 959kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<10:31, 1.20MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<07:40, 1.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:18, 1.52MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:20, 1.51MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:22, 1.98MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<04:35, 2.73MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<12:52, 974kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:17, 1.22MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:30, 1.67MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:09, 1.53MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:12, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:16, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:32<04:32, 2.74MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<10:15, 1.21MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:27, 1.47MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:12, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:14, 1.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<07:33, 1.63MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:49, 2.12MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:36<04:13, 2.91MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<10:17, 1.19MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<08:26, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:12, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<07:12, 1.69MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<07:29, 1.63MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:50, 2.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<04:11, 2.89MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<50:14, 242kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<36:23, 333kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<25:44, 470kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<20:46, 581kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<16:57, 711kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<12:28, 966kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<10:38, 1.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<09:57, 1.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<07:34, 1.58MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<05:25, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<1:29:16, 134kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<1:03:41, 187kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<44:43, 266kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<34:01, 349kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<26:11, 453kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<18:54, 626kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<15:05, 781kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<11:44, 1.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:51<08:27, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:39, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<08:25, 1.39MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<06:28, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<04:38, 2.51MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<11:17:08, 17.2kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<7:54:45, 24.5kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<5:31:54, 35.0kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<3:54:17, 49.4kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<2:46:17, 69.6kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<1:56:46, 99.1kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<1:21:34, 141kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:07:43, 170kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<48:31, 237kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<34:08, 336kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<26:32, 431kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<19:31, 586kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<13:51, 824kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<10:23, 1.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<8:39:46, 21.9kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<6:04:05, 31.2kB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:03<4:13:56, 44.6kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<3:07:38, 60.3kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<2:13:35, 84.7kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<1:33:57, 120kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<1:07:18, 167kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<49:27, 227kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<35:04, 320kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:07<24:36, 455kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<24:38, 454kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<18:23, 607kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<13:05, 852kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<11:44, 946kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<09:08, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<06:40, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:16, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:18, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:34, 1.98MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:13<04:01, 2.73MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<08:58, 1.22MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<07:24, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:24, 2.02MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:20, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<06:37, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:10, 2.10MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:22, 2.02MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:56, 1.83MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<04:37, 2.34MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:26, 3.14MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<05:47, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<04:59, 2.15MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<03:46, 2.85MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<05:07, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<05:44, 1.86MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<04:33, 2.35MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:17, 3.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<1:19:21, 134kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<56:34, 188kB/s]  .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<39:44, 267kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 228M/862M [01:26<30:13, 349kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<22:12, 475kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<15:44, 669kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<13:26, 780kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:29<11:31, 910kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<08:31, 1.23MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<06:05, 1.71MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<09:08, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<07:39, 1.36MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<05:39, 1.84MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:05, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:45, 1.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<05:15, 1.97MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<03:48, 2.71MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:52, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<06:02, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<04:32, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<05:15, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<06:08, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:37<04:50, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<03:33, 2.86MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:22, 1.89MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<04:46, 2.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<03:38, 2.78MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:37, 2.18MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:27, 2.26MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<03:21, 2.99MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:24, 2.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<05:30, 1.82MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:22, 2.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:10, 3.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:33, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:45, 1.72MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:17, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:01, 1.96MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:54, 1.67MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:38, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<03:25, 2.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<05:10, 1.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<04:48, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:39, 2.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:32, 2.14MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:32, 1.76MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<04:27, 2.18MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<03:15, 2.97MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<08:27, 1.14MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<07:04, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<05:11, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<05:36, 1.71MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<06:14, 1.54MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<04:55, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<03:33, 2.68MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:56<09:17, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<07:37, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<05:36, 1.69MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:51, 1.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<06:22, 1.48MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:56, 1.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:39, 2.57MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:50, 1.94MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:31, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:26, 2.72MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:18, 2.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:15, 1.77MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:09, 2.24MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:02<03:03, 3.02MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<04:50, 1.91MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<04:30, 2.05MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<03:25, 2.69MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:15, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:55, 2.34MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<02:58, 3.08MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:06<02:15, 4.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<07:15, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<05:57, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<04:26, 2.04MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<03:37, 2.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<6:27:40, 23.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<4:31:34, 33.3kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:10<3:09:18, 47.5kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<2:17:48, 65.1kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<1:38:36, 91.0kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<1:09:24, 129kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<48:35, 184kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<36:30, 244kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<26:36, 334kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<18:51, 471kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<14:56, 591kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<12:42, 695kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<09:19, 946kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:16<06:37, 1.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<07:57, 1.10MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:18<06:27, 1.36MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:46, 1.83MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:20<05:06, 1.70MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<05:40, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<04:25, 1.96MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:20<03:11, 2.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<06:19, 1.37MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:22<05:27, 1.58MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<04:04, 2.11MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:35, 1.86MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:24<04:14, 2.01MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:10, 2.68MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:57, 2.14MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<03:46, 2.25MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<02:53, 2.93MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:45, 2.24MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:28<03:39, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<02:46, 3.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:39, 2.29MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<04:34, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<03:37, 2.30MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<02:39, 3.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:32<04:35, 1.80MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:32<04:13, 1.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<03:09, 2.62MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:53, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<04:48, 1.70MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:34<03:47, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<02:46, 2.94MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:31, 1.80MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:08, 1.96MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<03:05, 2.62MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<03:49, 2.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:38<04:31, 1.78MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:37, 2.22MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:38<02:37, 3.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<08:51, 903kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:40<07:09, 1.12MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:11, 1.53MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:15, 1.51MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:42<05:41, 1.39MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<04:24, 1.80MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:42<03:09, 2.49MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:44<06:52, 1.14MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<05:45, 1.37MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<04:13, 1.86MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<04:32, 1.71MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<05:03, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<03:57, 1.97MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:46<02:50, 2.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:48<06:08, 1.26MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:48<05:13, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<03:52, 1.98MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:16, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:50<04:50, 1.58MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<03:46, 2.02MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:44, 2.77MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:45, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:52<04:14, 1.79MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:11, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<03:45, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:54<04:16, 1.75MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<03:24, 2.20MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:54<02:28, 3.01MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:56<10:46, 691kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<08:21, 891kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:00, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:47, 1.27MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<05:55, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<04:30, 1.63MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<03:20, 2.20MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:59, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<03:35, 2.03MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<02:42, 2.69MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<03:27, 2.09MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:02<03:17, 2.19MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<02:29, 2.90MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:12, 2.23MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:04<03:58, 1.80MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<03:09, 2.27MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<02:20, 3.05MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:34, 1.98MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<03:20, 2.12MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<02:30, 2.82MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:12, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:47, 1.86MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<03:02, 2.30MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:08<02:11, 3.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<09:01, 772kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<07:00, 992kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:06, 1.36MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<04:58, 1.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<05:08, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<03:58, 1.73MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:12<02:50, 2.40MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<05:38, 1.21MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<04:38, 1.47MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:26, 1.97MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:47, 1.78MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<03:22, 2.00MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<02:32, 2.65MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:14, 2.07MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<03:43, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<02:58, 2.24MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:08, 3.08MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<08:01, 824kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<06:21, 1.04MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:36, 1.43MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:37, 1.41MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<04:49, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<03:46, 1.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:42, 2.39MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:37, 1.15MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<04:43, 1.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<03:27, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:43, 1.72MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<04:09, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<03:13, 1.98MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:20, 2.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<04:11, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:40, 1.72MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:45, 2.29MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:12, 1.96MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:36, 1.73MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<02:52, 2.18MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<02:04, 3.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<08:06, 764kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<06:17, 984kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<04:34, 1.35MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:26, 1.38MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<04:35, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:32, 1.73MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:31, 2.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<05:06, 1.19MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:18, 1.41MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<03:11, 1.89MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:26, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:51, 1.55MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:37<03:04, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:12, 2.68MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:58, 1.19MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:07, 1.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:39<03:00, 1.95MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:22, 1.74MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<03:50, 1.52MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<02:58, 1.96MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:09, 2.69MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<03:50, 1.50MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:22, 1.71MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:31, 2.28MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:55, 1.95MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:26, 1.66MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:42, 2.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<01:58, 2.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:21, 1.68MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<02:55, 1.92MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<02:12, 2.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<02:41, 2.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:14, 1.72MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:49<02:36, 2.13MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<01:53, 2.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<04:48, 1.15MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:58, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:51<02:53, 1.89MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:11, 1.70MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:37, 1.50MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<02:48, 1.93MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<02:01, 2.66MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<03:26, 1.56MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:02, 1.77MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<02:16, 2.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:40, 1.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:30, 2.11MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:57<01:52, 2.80MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:23, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:09, 2.43MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<01:38, 3.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<01:28, 3.51MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<3:35:09, 24.0kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<2:30:34, 34.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<1:44:31, 48.8kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<1:16:44, 66.4kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<54:54, 92.7kB/s]  .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<38:37, 132kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:03<26:53, 187kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<21:13, 237kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<15:24, 326kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<10:51, 460kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<08:37, 574kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<07:13, 685kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<05:18, 932kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:07<03:46, 1.30MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<04:06, 1.19MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:27, 1.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<02:32, 1.91MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:45, 1.75MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<02:30, 1.92MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:51, 2.57MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:16, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:38, 1.80MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:03, 2.29MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:13<01:29, 3.14MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:13, 1.45MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:46, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:02, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:26, 1.89MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:08, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:17<01:32, 2.95MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<06:19, 717kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:55, 919kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:19<03:32, 1.27MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:25, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:23, 1.32MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:36, 1.70MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:52, 2.36MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<06:39, 661kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<05:07, 856kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<03:41, 1.18MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<03:30, 1.23MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:29, 1.24MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<02:39, 1.62MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<01:54, 2.24MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:10, 1.34MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:43, 1.56MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:00, 2.11MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:15, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:06, 1.99MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:33, 2.66MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:56, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:20, 1.75MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<01:53, 2.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:22, 2.96MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:33, 1.14MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:57, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<02:10, 1.85MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:19, 1.71MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:35, 1.54MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:00, 1.97MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:26, 2.72MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<03:05, 1.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<02:37, 1.48MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:56, 1.99MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:08, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:54, 2.01MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:24, 2.70MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:48, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:05, 1.80MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:40, 2.25MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:12, 3.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<05:15, 705kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<04:05, 905kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<02:55, 1.25MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:49, 1.28MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:51, 1.27MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<02:10, 1.66MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:33, 2.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<02:32, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:12, 1.62MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:37, 2.18MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<01:50, 1.89MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:10, 1.60MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:43, 2.01MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:14, 2.75MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<03:03, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:32, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:51, 1.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:19, 2.53MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<41:10, 81.5kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<29:32, 114kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<20:45, 161kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<14:29, 229kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<10:59, 299kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<07:59, 411kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<05:37, 580kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<03:57, 817kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<05:52, 547kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:53, 658kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<03:34, 896kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<02:32, 1.25MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:44, 1.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:17, 1.37MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:41, 1.85MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:48, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<02:00, 1.54MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:33, 1.98MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<01:07, 2.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:57, 1.53MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<01:44, 1.73MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:17, 2.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:29, 1.97MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<01:47, 1.64MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:23, 2.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:01, 2.83MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:32, 1.86MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<01:24, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:04, 2.66MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:18, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<01:35, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:17, 2.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:55, 2.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:22, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:57, 1.39MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:25, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:33, 1.70MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:12<01:40, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:18, 2.01MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:56, 2.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<03:48, 681kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<02:58, 871kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:08, 1.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<01:59, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:00, 1.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:31, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:05, 2.26MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:30, 1.62MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:21, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:00, 2.42MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:10, 2.03MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:06, 2.14MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:50, 2.81MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:03, 2.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:01, 2.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:44, 3.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:24, 1.60MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<01:15, 1.79MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<00:56, 2.37MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:05, 2.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:17, 1.69MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:00, 2.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:44, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:04, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:00, 2.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:44, 2.78MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:56, 2.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:53, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:40, 2.95MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:52, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:05, 1.81MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:52, 2.23MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:37, 3.06MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:37, 1.17MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:20, 1.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<00:58, 1.91MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:04, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:12, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:56, 1.94MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:40, 2.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:09, 1.51MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:59, 1.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:44, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:50, 1.99MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:47, 2.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:35, 2.81MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:44, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:54, 1.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<00:43, 2.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<00:31, 3.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:21, 1.15MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:07, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:49, 1.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:51, 1.72MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:45, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:33, 2.57MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:41, 2.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:49, 1.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:39, 2.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:48<00:28, 2.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:11, 1.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:59, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:43, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:44, 1.70MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:49, 1.53MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:38, 1.97MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:27, 2.70MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:27, 2.61MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<47:17, 25.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<32:32, 36.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<22:05, 51.8kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<15:42, 72.6kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<10:56, 103kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<07:26, 147kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<05:30, 195kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<03:58, 269kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<02:45, 380kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<02:03, 489kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:41, 593kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<01:13, 812kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:50, 1.14MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:55, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:44, 1.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:31, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:32, 1.59MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:34, 1.52MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:26, 1.94MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:04<00:18, 2.66MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:10, 676kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:53, 887kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:37, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:34, 1.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:29, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:20, 2.02MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:21, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:23, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.12MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:12, 2.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:19, 1.81MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:17, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:12, 2.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:14, 2.11MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:17, 1.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:13, 2.17MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.97MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:23, 1.16MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:18, 1.41MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:13, 1.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:13, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.59MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:10, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:06, 2.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:26, 710kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:20, 903kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:13, 1.25MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:20<00:08, 1.74MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:45, 320kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:34, 411kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:23, 570kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:15, 802kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:11, 873kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:09, 1.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:05, 1.50MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.82MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.49MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:01, 1.58MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:00, 1.78MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 2.36MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<25:06:59,  4.42it/s]  0%|          | 907/400000 [00:00<17:32:43,  6.32it/s]  0%|          | 1855/400000 [00:00<12:15:21,  9.02it/s]  1%|          | 2712/400000 [00:00<8:33:52, 12.89it/s]   1%|          | 3603/400000 [00:00<5:59:07, 18.40it/s]  1%|          | 4417/400000 [00:00<4:11:07, 26.25it/s]  1%|▏         | 5360/400000 [00:00<2:55:34, 37.46it/s]  2%|▏         | 6289/400000 [00:00<2:02:49, 53.42it/s]  2%|▏         | 7221/400000 [00:01<1:25:59, 76.13it/s]  2%|▏         | 8140/400000 [00:01<1:00:15, 108.38it/s]  2%|▏         | 9132/400000 [00:01<42:16, 154.10it/s]    3%|▎         | 10044/400000 [00:01<29:44, 218.53it/s]  3%|▎         | 10993/400000 [00:01<20:58, 309.13it/s]  3%|▎         | 11965/400000 [00:01<14:50, 435.67it/s]  3%|▎         | 12950/400000 [00:01<10:33, 610.80it/s]  3%|▎         | 13912/400000 [00:01<07:34, 849.44it/s]  4%|▎         | 14876/400000 [00:01<05:29, 1169.32it/s]  4%|▍         | 15833/400000 [00:01<04:02, 1583.75it/s]  4%|▍         | 16777/400000 [00:02<03:01, 2107.23it/s]  4%|▍         | 17748/400000 [00:02<02:18, 2754.02it/s]  5%|▍         | 18702/400000 [00:02<01:48, 3500.72it/s]  5%|▍         | 19659/400000 [00:02<01:27, 4322.99it/s]  5%|▌         | 20611/400000 [00:02<01:13, 5165.93it/s]  5%|▌         | 21562/400000 [00:02<01:04, 5884.44it/s]  6%|▌         | 22489/400000 [00:02<00:57, 6606.33it/s]  6%|▌         | 23437/400000 [00:02<00:51, 7266.16it/s]  6%|▌         | 24404/400000 [00:02<00:47, 7851.42it/s]  6%|▋         | 25374/400000 [00:02<00:44, 8325.77it/s]  7%|▋         | 26325/400000 [00:03<00:43, 8643.26it/s]  7%|▋         | 27314/400000 [00:03<00:41, 8981.19it/s]  7%|▋         | 28276/400000 [00:03<00:40, 9153.05it/s]  7%|▋         | 29255/400000 [00:03<00:39, 9332.62it/s]  8%|▊         | 30250/400000 [00:03<00:38, 9507.83it/s]  8%|▊         | 31225/400000 [00:03<00:38, 9523.82it/s]  8%|▊         | 32194/400000 [00:03<00:38, 9570.89it/s]  8%|▊         | 33163/400000 [00:03<00:38, 9584.95it/s]  9%|▊         | 34130/400000 [00:03<00:38, 9599.14it/s]  9%|▉         | 35121/400000 [00:03<00:37, 9689.06it/s]  9%|▉         | 36095/400000 [00:04<00:38, 9460.68it/s]  9%|▉         | 37072/400000 [00:04<00:38, 9549.88it/s] 10%|▉         | 38042/400000 [00:04<00:37, 9592.01it/s] 10%|▉         | 39004/400000 [00:04<00:37, 9515.75it/s] 10%|▉         | 39962/400000 [00:04<00:37, 9532.73it/s] 10%|█         | 40917/400000 [00:04<00:38, 9330.23it/s] 10%|█         | 41860/400000 [00:04<00:38, 9358.39it/s] 11%|█         | 42798/400000 [00:04<00:38, 9304.70it/s] 11%|█         | 43730/400000 [00:04<00:38, 9237.51it/s] 11%|█         | 44710/400000 [00:04<00:37, 9397.70it/s] 11%|█▏        | 45651/400000 [00:05<00:37, 9388.81it/s] 12%|█▏        | 46638/400000 [00:05<00:37, 9526.50it/s] 12%|█▏        | 47637/400000 [00:05<00:36, 9660.46it/s] 12%|█▏        | 48607/400000 [00:05<00:36, 9671.43it/s] 12%|█▏        | 49606/400000 [00:05<00:35, 9764.51it/s] 13%|█▎        | 50584/400000 [00:05<00:36, 9587.46it/s] 13%|█▎        | 51545/400000 [00:05<00:36, 9428.30it/s] 13%|█▎        | 52500/400000 [00:05<00:36, 9462.92it/s] 13%|█▎        | 53450/400000 [00:05<00:36, 9472.57it/s] 14%|█▎        | 54399/400000 [00:05<00:36, 9431.68it/s] 14%|█▍        | 55343/400000 [00:06<00:37, 9285.72it/s] 14%|█▍        | 56273/400000 [00:06<00:37, 9218.56it/s] 14%|█▍        | 57196/400000 [00:06<00:37, 9212.45it/s] 15%|█▍        | 58170/400000 [00:06<00:36, 9363.20it/s] 15%|█▍        | 59108/400000 [00:06<00:36, 9302.49it/s] 15%|█▌        | 60056/400000 [00:06<00:36, 9354.50it/s] 15%|█▌        | 61045/400000 [00:06<00:35, 9508.52it/s] 16%|█▌        | 62033/400000 [00:06<00:35, 9614.90it/s] 16%|█▌        | 62999/400000 [00:06<00:35, 9628.23it/s] 16%|█▌        | 63983/400000 [00:07<00:34, 9689.46it/s] 16%|█▌        | 64958/400000 [00:07<00:34, 9707.26it/s] 16%|█▋        | 65930/400000 [00:07<00:34, 9680.92it/s] 17%|█▋        | 66930/400000 [00:07<00:34, 9772.91it/s] 17%|█▋        | 67915/400000 [00:07<00:33, 9795.81it/s] 17%|█▋        | 68898/400000 [00:07<00:33, 9803.98it/s] 17%|█▋        | 69879/400000 [00:07<00:34, 9677.65it/s] 18%|█▊        | 70856/400000 [00:07<00:33, 9703.87it/s] 18%|█▊        | 71827/400000 [00:07<00:33, 9679.64it/s] 18%|█▊        | 72811/400000 [00:07<00:33, 9726.87it/s] 18%|█▊        | 73784/400000 [00:08<00:33, 9708.46it/s] 19%|█▊        | 74756/400000 [00:08<00:34, 9405.31it/s] 19%|█▉        | 75725/400000 [00:08<00:34, 9488.30it/s] 19%|█▉        | 76676/400000 [00:08<00:34, 9489.56it/s] 19%|█▉        | 77653/400000 [00:08<00:33, 9570.59it/s] 20%|█▉        | 78631/400000 [00:08<00:33, 9629.62it/s] 20%|█▉        | 79595/400000 [00:08<00:34, 9314.28it/s] 20%|██        | 80548/400000 [00:08<00:34, 9375.57it/s] 20%|██        | 81504/400000 [00:08<00:33, 9428.69it/s] 21%|██        | 82449/400000 [00:08<00:34, 9317.02it/s] 21%|██        | 83383/400000 [00:09<00:34, 9215.86it/s] 21%|██        | 84306/400000 [00:09<00:34, 9129.72it/s] 21%|██▏       | 85254/400000 [00:09<00:34, 9231.41it/s] 22%|██▏       | 86221/400000 [00:09<00:33, 9358.74it/s] 22%|██▏       | 87158/400000 [00:09<00:33, 9298.11it/s] 22%|██▏       | 88089/400000 [00:09<00:33, 9241.74it/s] 22%|██▏       | 89014/400000 [00:09<00:33, 9243.94it/s] 22%|██▏       | 89967/400000 [00:09<00:33, 9325.50it/s] 23%|██▎       | 90901/400000 [00:09<00:33, 9210.85it/s] 23%|██▎       | 91823/400000 [00:09<00:33, 9178.36it/s] 23%|██▎       | 92791/400000 [00:10<00:32, 9320.85it/s] 23%|██▎       | 93724/400000 [00:10<00:33, 9133.67it/s] 24%|██▎       | 94692/400000 [00:10<00:32, 9288.92it/s] 24%|██▍       | 95623/400000 [00:10<00:33, 9166.88it/s] 24%|██▍       | 96542/400000 [00:10<00:34, 8864.31it/s] 24%|██▍       | 97432/400000 [00:10<00:35, 8618.98it/s] 25%|██▍       | 98350/400000 [00:10<00:34, 8778.78it/s] 25%|██▍       | 99269/400000 [00:10<00:33, 8896.92it/s] 25%|██▌       | 100189/400000 [00:10<00:33, 8982.96it/s] 25%|██▌       | 101110/400000 [00:10<00:33, 9048.04it/s] 26%|██▌       | 102044/400000 [00:11<00:32, 9131.65it/s] 26%|██▌       | 102979/400000 [00:11<00:32, 9195.79it/s] 26%|██▌       | 103961/400000 [00:11<00:31, 9373.56it/s] 26%|██▌       | 104919/400000 [00:11<00:31, 9429.30it/s] 26%|██▋       | 105864/400000 [00:11<00:32, 9163.95it/s] 27%|██▋       | 106826/400000 [00:11<00:31, 9294.43it/s] 27%|██▋       | 107762/400000 [00:11<00:31, 9309.04it/s] 27%|██▋       | 108748/400000 [00:11<00:30, 9466.29it/s] 27%|██▋       | 109723/400000 [00:11<00:30, 9548.11it/s] 28%|██▊       | 110680/400000 [00:11<00:30, 9445.65it/s] 28%|██▊       | 111670/400000 [00:12<00:30, 9575.96it/s] 28%|██▊       | 112629/400000 [00:12<00:30, 9318.16it/s] 28%|██▊       | 113591/400000 [00:12<00:30, 9404.19it/s] 29%|██▊       | 114545/400000 [00:12<00:30, 9442.62it/s] 29%|██▉       | 115538/400000 [00:12<00:29, 9583.49it/s] 29%|██▉       | 116498/400000 [00:12<00:29, 9506.42it/s] 29%|██▉       | 117450/400000 [00:12<00:29, 9439.16it/s] 30%|██▉       | 118400/400000 [00:12<00:29, 9456.28it/s] 30%|██▉       | 119353/400000 [00:12<00:29, 9476.51it/s] 30%|███       | 120313/400000 [00:13<00:29, 9512.31it/s] 30%|███       | 121294/400000 [00:13<00:29, 9599.45it/s] 31%|███       | 122255/400000 [00:13<00:29, 9549.37it/s] 31%|███       | 123245/400000 [00:13<00:28, 9649.72it/s] 31%|███       | 124245/400000 [00:13<00:28, 9749.66it/s] 31%|███▏      | 125221/400000 [00:13<00:28, 9747.51it/s] 32%|███▏      | 126197/400000 [00:13<00:28, 9749.45it/s] 32%|███▏      | 127173/400000 [00:13<00:28, 9414.45it/s] 32%|███▏      | 128118/400000 [00:13<00:28, 9423.40it/s] 32%|███▏      | 129063/400000 [00:13<00:29, 9320.31it/s] 33%|███▎      | 130011/400000 [00:14<00:28, 9365.64it/s] 33%|███▎      | 130949/400000 [00:14<00:28, 9328.01it/s] 33%|███▎      | 131891/400000 [00:14<00:28, 9355.16it/s] 33%|███▎      | 132828/400000 [00:14<00:29, 9173.00it/s] 33%|███▎      | 133789/400000 [00:14<00:28, 9298.47it/s] 34%|███▎      | 134721/400000 [00:14<00:28, 9227.51it/s] 34%|███▍      | 135645/400000 [00:14<00:29, 9059.97it/s] 34%|███▍      | 136553/400000 [00:14<00:29, 9032.00it/s] 34%|███▍      | 137472/400000 [00:14<00:28, 9077.78it/s] 35%|███▍      | 138381/400000 [00:14<00:28, 9055.77it/s] 35%|███▍      | 139288/400000 [00:15<00:28, 9016.93it/s] 35%|███▌      | 140215/400000 [00:15<00:28, 9090.92it/s] 35%|███▌      | 141172/400000 [00:15<00:28, 9229.28it/s] 36%|███▌      | 142139/400000 [00:15<00:27, 9356.13it/s] 36%|███▌      | 143076/400000 [00:15<00:27, 9279.64it/s] 36%|███▌      | 144005/400000 [00:15<00:27, 9187.79it/s] 36%|███▌      | 144925/400000 [00:15<00:28, 9073.93it/s] 36%|███▋      | 145834/400000 [00:15<00:28, 9015.00it/s] 37%|███▋      | 146737/400000 [00:15<00:28, 9006.04it/s] 37%|███▋      | 147639/400000 [00:15<00:28, 8805.98it/s] 37%|███▋      | 148521/400000 [00:16<00:28, 8765.99it/s] 37%|███▋      | 149399/400000 [00:16<00:28, 8693.95it/s] 38%|███▊      | 150277/400000 [00:16<00:28, 8717.37it/s] 38%|███▊      | 151275/400000 [00:16<00:27, 9060.84it/s] 38%|███▊      | 152199/400000 [00:16<00:27, 9111.92it/s] 38%|███▊      | 153190/400000 [00:16<00:26, 9335.73it/s] 39%|███▊      | 154127/400000 [00:16<00:26, 9212.68it/s] 39%|███▉      | 155068/400000 [00:16<00:26, 9268.70it/s] 39%|███▉      | 155997/400000 [00:16<00:26, 9272.66it/s] 39%|███▉      | 156926/400000 [00:16<00:26, 9116.25it/s] 39%|███▉      | 157840/400000 [00:17<00:26, 9082.09it/s] 40%|███▉      | 158750/400000 [00:17<00:26, 8963.30it/s] 40%|███▉      | 159678/400000 [00:17<00:26, 9054.68it/s] 40%|████      | 160642/400000 [00:17<00:25, 9221.78it/s] 40%|████      | 161571/400000 [00:17<00:25, 9239.95it/s] 41%|████      | 162497/400000 [00:17<00:25, 9202.87it/s] 41%|████      | 163419/400000 [00:17<00:26, 9090.65it/s] 41%|████      | 164329/400000 [00:17<00:26, 9003.29it/s] 41%|████▏     | 165272/400000 [00:17<00:25, 9124.97it/s] 42%|████▏     | 166214/400000 [00:18<00:25, 9209.37it/s] 42%|████▏     | 167136/400000 [00:18<00:25, 9015.16it/s] 42%|████▏     | 168040/400000 [00:18<00:25, 8973.48it/s] 42%|████▏     | 168939/400000 [00:18<00:26, 8870.47it/s] 42%|████▏     | 169828/400000 [00:18<00:25, 8864.44it/s] 43%|████▎     | 170740/400000 [00:18<00:25, 8937.69it/s] 43%|████▎     | 171635/400000 [00:18<00:25, 8892.10it/s] 43%|████▎     | 172525/400000 [00:18<00:25, 8783.53it/s] 43%|████▎     | 173430/400000 [00:18<00:25, 8861.60it/s] 44%|████▎     | 174335/400000 [00:18<00:25, 8915.02it/s] 44%|████▍     | 175257/400000 [00:19<00:24, 9003.17it/s] 44%|████▍     | 176176/400000 [00:19<00:24, 9057.14it/s] 44%|████▍     | 177135/400000 [00:19<00:24, 9208.76it/s] 45%|████▍     | 178057/400000 [00:19<00:24, 9201.36it/s] 45%|████▍     | 178993/400000 [00:19<00:23, 9247.31it/s] 45%|████▍     | 179950/400000 [00:19<00:23, 9340.21it/s] 45%|████▌     | 180885/400000 [00:19<00:23, 9238.87it/s] 45%|████▌     | 181822/400000 [00:19<00:23, 9277.29it/s] 46%|████▌     | 182751/400000 [00:19<00:23, 9196.32it/s] 46%|████▌     | 183744/400000 [00:19<00:22, 9402.96it/s] 46%|████▌     | 184733/400000 [00:20<00:22, 9541.31it/s] 46%|████▋     | 185696/400000 [00:20<00:22, 9566.68it/s] 47%|████▋     | 186669/400000 [00:20<00:22, 9613.24it/s] 47%|████▋     | 187635/400000 [00:20<00:22, 9624.54it/s] 47%|████▋     | 188599/400000 [00:20<00:23, 9115.74it/s] 47%|████▋     | 189517/400000 [00:20<00:23, 9041.48it/s] 48%|████▊     | 190500/400000 [00:20<00:22, 9263.46it/s] 48%|████▊     | 191443/400000 [00:20<00:22, 9312.19it/s] 48%|████▊     | 192411/400000 [00:20<00:22, 9417.58it/s] 48%|████▊     | 193391/400000 [00:20<00:21, 9526.77it/s] 49%|████▊     | 194346/400000 [00:21<00:21, 9464.26it/s] 49%|████▉     | 195321/400000 [00:21<00:21, 9547.97it/s] 49%|████▉     | 196302/400000 [00:21<00:21, 9620.46it/s] 49%|████▉     | 197266/400000 [00:21<00:21, 9489.13it/s] 50%|████▉     | 198219/400000 [00:21<00:21, 9501.16it/s] 50%|████▉     | 199196/400000 [00:21<00:20, 9578.28it/s] 50%|█████     | 200202/400000 [00:21<00:20, 9715.73it/s] 50%|█████     | 201196/400000 [00:21<00:20, 9780.22it/s] 51%|█████     | 202175/400000 [00:21<00:20, 9682.67it/s] 51%|█████     | 203145/400000 [00:21<00:20, 9485.25it/s] 51%|█████     | 204095/400000 [00:22<00:20, 9468.28it/s] 51%|█████▏    | 205043/400000 [00:22<00:21, 9206.74it/s] 52%|█████▏    | 206009/400000 [00:22<00:20, 9336.13it/s] 52%|█████▏    | 206972/400000 [00:22<00:20, 9420.16it/s] 52%|█████▏    | 207955/400000 [00:22<00:20, 9536.67it/s] 52%|█████▏    | 208911/400000 [00:22<00:20, 9395.78it/s] 52%|█████▏    | 209853/400000 [00:22<00:20, 9351.00it/s] 53%|█████▎    | 210798/400000 [00:22<00:20, 9380.02it/s] 53%|█████▎    | 211768/400000 [00:22<00:19, 9472.13it/s] 53%|█████▎    | 212765/400000 [00:22<00:19, 9611.94it/s] 53%|█████▎    | 213728/400000 [00:23<00:19, 9530.38it/s] 54%|█████▎    | 214710/400000 [00:23<00:19, 9615.36it/s] 54%|█████▍    | 215707/400000 [00:23<00:18, 9716.15it/s] 54%|█████▍    | 216680/400000 [00:23<00:19, 9292.17it/s] 54%|█████▍    | 217646/400000 [00:23<00:19, 9397.22it/s] 55%|█████▍    | 218590/400000 [00:23<00:19, 9380.27it/s] 55%|█████▍    | 219540/400000 [00:23<00:19, 9413.91it/s] 55%|█████▌    | 220484/400000 [00:23<00:19, 9357.00it/s] 55%|█████▌    | 221433/400000 [00:23<00:19, 9395.08it/s] 56%|█████▌    | 222408/400000 [00:24<00:18, 9495.86it/s] 56%|█████▌    | 223382/400000 [00:24<00:18, 9565.19it/s] 56%|█████▌    | 224340/400000 [00:24<00:18, 9562.82it/s] 56%|█████▋    | 225297/400000 [00:24<00:18, 9432.82it/s] 57%|█████▋    | 226242/400000 [00:24<00:18, 9387.72it/s] 57%|█████▋    | 227190/400000 [00:24<00:18, 9414.85it/s] 57%|█████▋    | 228132/400000 [00:24<00:18, 9378.69it/s] 57%|█████▋    | 229071/400000 [00:24<00:18, 9304.54it/s] 58%|█████▊    | 230002/400000 [00:24<00:18, 9103.76it/s] 58%|█████▊    | 230943/400000 [00:24<00:18, 9192.51it/s] 58%|█████▊    | 231930/400000 [00:25<00:17, 9385.05it/s] 58%|█████▊    | 232897/400000 [00:25<00:17, 9467.15it/s] 58%|█████▊    | 233846/400000 [00:25<00:17, 9340.15it/s] 59%|█████▊    | 234782/400000 [00:25<00:17, 9247.30it/s] 59%|█████▉    | 235735/400000 [00:25<00:17, 9327.11it/s] 59%|█████▉    | 236694/400000 [00:25<00:17, 9402.56it/s] 59%|█████▉    | 237636/400000 [00:25<00:17, 9380.70it/s] 60%|█████▉    | 238619/400000 [00:25<00:16, 9509.78it/s] 60%|█████▉    | 239571/400000 [00:25<00:17, 9407.12it/s] 60%|██████    | 240513/400000 [00:25<00:17, 9340.61it/s] 60%|██████    | 241448/400000 [00:26<00:17, 9024.45it/s] 61%|██████    | 242362/400000 [00:26<00:17, 9056.09it/s] 61%|██████    | 243322/400000 [00:26<00:17, 9212.52it/s] 61%|██████    | 244296/400000 [00:26<00:16, 9362.88it/s] 61%|██████▏   | 245249/400000 [00:26<00:16, 9410.35it/s] 62%|██████▏   | 246192/400000 [00:26<00:17, 8893.91it/s] 62%|██████▏   | 247096/400000 [00:26<00:17, 8936.32it/s] 62%|██████▏   | 247995/400000 [00:26<00:17, 8799.19it/s] 62%|██████▏   | 248904/400000 [00:26<00:17, 8884.18it/s] 62%|██████▏   | 249859/400000 [00:26<00:16, 9072.41it/s] 63%|██████▎   | 250792/400000 [00:27<00:16, 9145.56it/s] 63%|██████▎   | 251709/400000 [00:27<00:16, 9025.77it/s] 63%|██████▎   | 252666/400000 [00:27<00:16, 9180.78it/s] 63%|██████▎   | 253597/400000 [00:27<00:15, 9218.70it/s] 64%|██████▎   | 254566/400000 [00:27<00:15, 9353.96it/s] 64%|██████▍   | 255530/400000 [00:27<00:15, 9435.83it/s] 64%|██████▍   | 256512/400000 [00:27<00:15, 9546.52it/s] 64%|██████▍   | 257468/400000 [00:27<00:15, 9271.48it/s] 65%|██████▍   | 258398/400000 [00:27<00:15, 9182.47it/s] 65%|██████▍   | 259334/400000 [00:28<00:15, 9233.65it/s] 65%|██████▌   | 260267/400000 [00:28<00:15, 9260.92it/s] 65%|██████▌   | 261195/400000 [00:28<00:15, 9183.49it/s] 66%|██████▌   | 262143/400000 [00:28<00:14, 9268.32it/s] 66%|██████▌   | 263071/400000 [00:28<00:15, 9099.98it/s] 66%|██████▌   | 263983/400000 [00:28<00:15, 9067.49it/s] 66%|██████▌   | 264930/400000 [00:28<00:14, 9183.55it/s] 66%|██████▋   | 265897/400000 [00:28<00:14, 9323.42it/s] 67%|██████▋   | 266831/400000 [00:28<00:14, 9319.55it/s] 67%|██████▋   | 267764/400000 [00:28<00:14, 9047.31it/s] 67%|██████▋   | 268672/400000 [00:29<00:14, 8826.91it/s] 67%|██████▋   | 269565/400000 [00:29<00:14, 8855.11it/s] 68%|██████▊   | 270465/400000 [00:29<00:14, 8895.81it/s] 68%|██████▊   | 271443/400000 [00:29<00:14, 9143.65it/s] 68%|██████▊   | 272361/400000 [00:29<00:14, 9101.04it/s] 68%|██████▊   | 273296/400000 [00:29<00:13, 9173.76it/s] 69%|██████▊   | 274215/400000 [00:29<00:13, 9016.51it/s] 69%|██████▉   | 275127/400000 [00:29<00:13, 9045.16it/s] 69%|██████▉   | 276033/400000 [00:29<00:13, 8997.90it/s] 69%|██████▉   | 276941/400000 [00:29<00:13, 9020.42it/s] 69%|██████▉   | 277888/400000 [00:30<00:13, 9149.22it/s] 70%|██████▉   | 278847/400000 [00:30<00:13, 9274.98it/s] 70%|██████▉   | 279839/400000 [00:30<00:12, 9456.81it/s] 70%|███████   | 280799/400000 [00:30<00:12, 9497.77it/s] 70%|███████   | 281750/400000 [00:30<00:12, 9254.33it/s] 71%|███████   | 282696/400000 [00:30<00:12, 9313.47it/s] 71%|███████   | 283675/400000 [00:30<00:12, 9448.77it/s] 71%|███████   | 284622/400000 [00:30<00:12, 9355.58it/s] 71%|███████▏  | 285559/400000 [00:30<00:12, 9354.43it/s] 72%|███████▏  | 286496/400000 [00:30<00:12, 9165.52it/s] 72%|███████▏  | 287432/400000 [00:31<00:12, 9221.58it/s] 72%|███████▏  | 288391/400000 [00:31<00:11, 9326.27it/s] 72%|███████▏  | 289325/400000 [00:31<00:11, 9301.80it/s] 73%|███████▎  | 290285/400000 [00:31<00:11, 9385.78it/s] 73%|███████▎  | 291225/400000 [00:31<00:11, 9299.88it/s] 73%|███████▎  | 292156/400000 [00:31<00:11, 9031.53it/s] 73%|███████▎  | 293062/400000 [00:31<00:12, 8901.51it/s] 73%|███████▎  | 293955/400000 [00:31<00:12, 8771.54it/s] 74%|███████▎  | 294847/400000 [00:31<00:11, 8814.93it/s] 74%|███████▍  | 295766/400000 [00:31<00:11, 8921.36it/s] 74%|███████▍  | 296693/400000 [00:32<00:11, 9022.99it/s] 74%|███████▍  | 297597/400000 [00:32<00:11, 8897.78it/s] 75%|███████▍  | 298488/400000 [00:32<00:11, 8760.51it/s] 75%|███████▍  | 299411/400000 [00:32<00:11, 8895.06it/s] 75%|███████▌  | 300360/400000 [00:32<00:10, 9063.66it/s] 75%|███████▌  | 301287/400000 [00:32<00:10, 9123.81it/s] 76%|███████▌  | 302260/400000 [00:32<00:10, 9295.29it/s] 76%|███████▌  | 303224/400000 [00:32<00:10, 9394.69it/s] 76%|███████▌  | 304165/400000 [00:32<00:10, 9373.97it/s] 76%|███████▋  | 305104/400000 [00:33<00:10, 9370.19it/s] 77%|███████▋  | 306054/400000 [00:33<00:09, 9407.64it/s] 77%|███████▋  | 307019/400000 [00:33<00:09, 9477.07it/s] 77%|███████▋  | 307968/400000 [00:33<00:09, 9441.09it/s] 77%|███████▋  | 308920/400000 [00:33<00:09, 9463.50it/s] 77%|███████▋  | 309867/400000 [00:33<00:09, 9448.68it/s] 78%|███████▊  | 310813/400000 [00:33<00:09, 9449.95it/s] 78%|███████▊  | 311775/400000 [00:33<00:09, 9497.97it/s] 78%|███████▊  | 312767/400000 [00:33<00:09, 9620.75it/s] 78%|███████▊  | 313730/400000 [00:33<00:08, 9622.28it/s] 79%|███████▊  | 314693/400000 [00:34<00:08, 9554.36it/s] 79%|███████▉  | 315686/400000 [00:34<00:08, 9661.85it/s] 79%|███████▉  | 316653/400000 [00:34<00:08, 9416.21it/s] 79%|███████▉  | 317628/400000 [00:34<00:08, 9513.10it/s] 80%|███████▉  | 318605/400000 [00:34<00:08, 9587.59it/s] 80%|███████▉  | 319565/400000 [00:34<00:08, 9543.16it/s] 80%|████████  | 320545/400000 [00:34<00:08, 9617.81it/s] 80%|████████  | 321542/400000 [00:34<00:08, 9718.75it/s] 81%|████████  | 322515/400000 [00:34<00:08, 9632.11it/s] 81%|████████  | 323479/400000 [00:34<00:08, 9553.68it/s] 81%|████████  | 324437/400000 [00:35<00:07, 9560.57it/s] 81%|████████▏ | 325413/400000 [00:35<00:07, 9619.38it/s] 82%|████████▏ | 326395/400000 [00:35<00:07, 9678.32it/s] 82%|████████▏ | 327364/400000 [00:35<00:07, 9560.39it/s] 82%|████████▏ | 328321/400000 [00:35<00:07, 9495.03it/s] 82%|████████▏ | 329272/400000 [00:35<00:07, 9152.36it/s] 83%|████████▎ | 330233/400000 [00:35<00:07, 9282.70it/s] 83%|████████▎ | 331185/400000 [00:35<00:07, 9350.91it/s] 83%|████████▎ | 332142/400000 [00:35<00:07, 9413.55it/s] 83%|████████▎ | 333085/400000 [00:35<00:07, 9256.36it/s] 84%|████████▎ | 334013/400000 [00:36<00:07, 9118.20it/s] 84%|████████▎ | 334982/400000 [00:36<00:07, 9280.64it/s] 84%|████████▍ | 335912/400000 [00:36<00:07, 9005.16it/s] 84%|████████▍ | 336816/400000 [00:36<00:07, 8992.87it/s] 84%|████████▍ | 337805/400000 [00:36<00:06, 9244.31it/s] 85%|████████▍ | 338742/400000 [00:36<00:06, 9279.53it/s] 85%|████████▍ | 339731/400000 [00:36<00:06, 9451.68it/s] 85%|████████▌ | 340719/400000 [00:36<00:06, 9575.35it/s] 85%|████████▌ | 341680/400000 [00:36<00:06, 9583.86it/s] 86%|████████▌ | 342640/400000 [00:36<00:06, 9523.95it/s] 86%|████████▌ | 343594/400000 [00:37<00:05, 9487.33it/s] 86%|████████▌ | 344589/400000 [00:37<00:05, 9620.15it/s] 86%|████████▋ | 345570/400000 [00:37<00:05, 9675.97it/s] 87%|████████▋ | 346540/400000 [00:37<00:05, 9683.13it/s] 87%|████████▋ | 347533/400000 [00:37<00:05, 9754.45it/s] 87%|████████▋ | 348509/400000 [00:37<00:05, 9754.72it/s] 87%|████████▋ | 349485/400000 [00:37<00:05, 9737.97it/s] 88%|████████▊ | 350478/400000 [00:37<00:05, 9791.60it/s] 88%|████████▊ | 351458/400000 [00:37<00:04, 9763.92it/s] 88%|████████▊ | 352453/400000 [00:37<00:04, 9817.44it/s] 88%|████████▊ | 353435/400000 [00:38<00:04, 9783.30it/s] 89%|████████▊ | 354414/400000 [00:38<00:04, 9472.54it/s] 89%|████████▉ | 355364/400000 [00:38<00:04, 9436.42it/s] 89%|████████▉ | 356310/400000 [00:38<00:04, 9406.79it/s] 89%|████████▉ | 357252/400000 [00:38<00:04, 9265.48it/s] 90%|████████▉ | 358180/400000 [00:38<00:04, 9161.30it/s] 90%|████████▉ | 359115/400000 [00:38<00:04, 9215.97it/s] 90%|█████████ | 360038/400000 [00:38<00:04, 8989.59it/s] 90%|█████████ | 360949/400000 [00:38<00:04, 9022.69it/s] 90%|█████████ | 361853/400000 [00:39<00:04, 9005.67it/s] 91%|█████████ | 362756/400000 [00:39<00:04, 9011.44it/s] 91%|█████████ | 363658/400000 [00:39<00:04, 8843.18it/s] 91%|█████████ | 364547/400000 [00:39<00:04, 8856.71it/s] 91%|█████████▏| 365434/400000 [00:39<00:03, 8855.68it/s] 92%|█████████▏| 366331/400000 [00:39<00:03, 8886.93it/s] 92%|█████████▏| 367221/400000 [00:39<00:03, 8788.88it/s] 92%|█████████▏| 368121/400000 [00:39<00:03, 8848.03it/s] 92%|█████████▏| 369017/400000 [00:39<00:03, 8879.21it/s] 92%|█████████▏| 369931/400000 [00:39<00:03, 8953.09it/s] 93%|█████████▎| 370871/400000 [00:40<00:03, 9080.88it/s] 93%|█████████▎| 371799/400000 [00:40<00:03, 9139.56it/s] 93%|█████████▎| 372714/400000 [00:40<00:03, 9012.49it/s] 93%|█████████▎| 373621/400000 [00:40<00:02, 9027.88it/s] 94%|█████████▎| 374525/400000 [00:40<00:02, 9019.87it/s] 94%|█████████▍| 375432/400000 [00:40<00:02, 9034.06it/s] 94%|█████████▍| 376336/400000 [00:40<00:02, 9010.71it/s] 94%|█████████▍| 377260/400000 [00:40<00:02, 9077.77it/s] 95%|█████████▍| 378169/400000 [00:40<00:02, 8854.70it/s] 95%|█████████▍| 379056/400000 [00:40<00:02, 8847.80it/s] 95%|█████████▍| 379978/400000 [00:41<00:02, 8954.43it/s] 95%|█████████▌| 380936/400000 [00:41<00:02, 9131.70it/s] 95%|█████████▌| 381903/400000 [00:41<00:01, 9284.55it/s] 96%|█████████▌| 382854/400000 [00:41<00:01, 9350.03it/s] 96%|█████████▌| 383791/400000 [00:41<00:01, 9303.89it/s] 96%|█████████▌| 384723/400000 [00:41<00:01, 9146.11it/s] 96%|█████████▋| 385639/400000 [00:41<00:01, 8980.28it/s] 97%|█████████▋| 386595/400000 [00:41<00:01, 9145.98it/s] 97%|█████████▋| 387560/400000 [00:41<00:01, 9290.52it/s] 97%|█████████▋| 388491/400000 [00:41<00:01, 9295.52it/s] 97%|█████████▋| 389422/400000 [00:42<00:01, 9228.56it/s] 98%|█████████▊| 390346/400000 [00:42<00:01, 8933.61it/s] 98%|█████████▊| 391243/400000 [00:42<00:01, 8623.54it/s] 98%|█████████▊| 392170/400000 [00:42<00:00, 8805.73it/s] 98%|█████████▊| 393090/400000 [00:42<00:00, 8917.94it/s] 99%|█████████▊| 394019/400000 [00:42<00:00, 9026.29it/s] 99%|█████████▊| 394925/400000 [00:42<00:00, 9022.99it/s] 99%|█████████▉| 395886/400000 [00:42<00:00, 9190.07it/s] 99%|█████████▉| 396807/400000 [00:42<00:00, 9110.13it/s] 99%|█████████▉| 397729/400000 [00:42<00:00, 9139.85it/s]100%|█████████▉| 398681/400000 [00:43<00:00, 9250.28it/s]100%|█████████▉| 399637/400000 [00:43<00:00, 9340.64it/s]100%|█████████▉| 399999/400000 [00:43<00:00, 9255.69it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fc441d73d30>, <torchtext.data.dataset.TabularDataset object at 0x7fc441d73e80>, <torchtext.vocab.Vocab object at 0x7fc441d73da0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  3.71 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  3.71 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  3.71 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.39 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.85 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.63 file/s]2020-07-27 00:17:31.818206: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-27 00:17:31.822264: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095195000 Hz
2020-07-27 00:17:31.822489: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56429c58e240 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-27 00:17:31.822510: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:14, 133543.75it/s] 65%|██████▌   | 6479872/9912422 [00:00<00:18, 190607.65it/s]9920512it [00:00, 39481885.40it/s]                           
0it [00:00, ?it/s]32768it [00:00, 540844.85it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 476404.73it/s]1654784it [00:00, 11751254.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 190736.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

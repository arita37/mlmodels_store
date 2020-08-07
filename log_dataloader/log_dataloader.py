
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f01167eb598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f01167eb598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f018149b488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f018149b488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f01a06b9ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f01a06b9ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f012e7d7158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f012e7d7158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f012e7d7158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 37%|███▋      | 3645440/9912422 [00:00<00:00, 36302794.45it/s]9920512it [00:00, 32761254.28it/s]                             
0it [00:00, ?it/s]32768it [00:00, 549204.41it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:12, 132559.44it/s]1654784it [00:00, 9413703.56it/s]                          
0it [00:00, ?it/s]8192it [00:00, 178721.47it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f012cbbe6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f012ccbd860>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f012e7cbd90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f012e7cbd90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f012e7cbd90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:26,  1.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:25,  1.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:25,  1.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:24,  1.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:24,  1.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:23,  1.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:58,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:58,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:57,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:57,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:57,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:56,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:56,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:55,  2.61 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:38,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:38,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:38,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:38,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:37,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:37,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:37,  3.68 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:26,  5.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:26,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:26,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:25,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:24,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:24,  5.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:17,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:17,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:17,  7.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:17,  7.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:17,  7.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:16,  7.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:16,  7.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:16,  7.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11,  9.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 13.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:08, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:08, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 18.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 23.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 29.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 37.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 37.02 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 44.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 52.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 65.13 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.84 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 80.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.22 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 80.22 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.47s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 80.22 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.60 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.97s/ url]
0 examples [00:00, ? examples/s]2020-08-07 18:07:39.374022: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-07 18:07:39.385046: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-07 18:07:39.385273: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fede8d1460 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-07 18:07:39.385294: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
32 examples [00:00, 318.67 examples/s]125 examples [00:00, 396.54 examples/s]216 examples [00:00, 476.71 examples/s]308 examples [00:00, 556.59 examples/s]397 examples [00:00, 626.14 examples/s]488 examples [00:00, 689.35 examples/s]581 examples [00:00, 746.33 examples/s]668 examples [00:00, 778.78 examples/s]759 examples [00:00, 812.74 examples/s]847 examples [00:01, 830.23 examples/s]940 examples [00:01, 856.85 examples/s]1032 examples [00:01, 868.30 examples/s]1123 examples [00:01, 878.55 examples/s]1215 examples [00:01, 889.31 examples/s]1308 examples [00:01, 898.90 examples/s]1399 examples [00:01, 886.16 examples/s]1490 examples [00:01, 891.29 examples/s]1583 examples [00:01, 900.16 examples/s]1675 examples [00:01, 903.40 examples/s]1766 examples [00:02, 904.47 examples/s]1857 examples [00:02, 904.17 examples/s]1948 examples [00:02, 902.26 examples/s]2039 examples [00:02, 881.47 examples/s]2128 examples [00:02, 870.18 examples/s]2216 examples [00:02, 859.24 examples/s]2309 examples [00:02, 877.94 examples/s]2404 examples [00:02, 895.67 examples/s]2494 examples [00:02, 886.09 examples/s]2583 examples [00:02, 882.93 examples/s]2673 examples [00:03, 886.31 examples/s]2762 examples [00:03, 886.76 examples/s]2853 examples [00:03, 891.35 examples/s]2943 examples [00:03, 870.38 examples/s]3034 examples [00:03, 881.42 examples/s]3127 examples [00:03, 893.72 examples/s]3217 examples [00:03, 893.61 examples/s]3307 examples [00:03, 894.96 examples/s]3398 examples [00:03, 898.51 examples/s]3489 examples [00:03, 901.76 examples/s]3580 examples [00:04, 902.49 examples/s]3671 examples [00:04, 856.83 examples/s]3763 examples [00:04, 872.49 examples/s]3853 examples [00:04, 879.25 examples/s]3946 examples [00:04, 891.99 examples/s]4037 examples [00:04, 896.57 examples/s]4127 examples [00:04, 891.97 examples/s]4220 examples [00:04, 901.67 examples/s]4311 examples [00:04, 900.81 examples/s]4402 examples [00:04, 901.71 examples/s]4494 examples [00:05, 905.05 examples/s]4585 examples [00:05, 901.77 examples/s]4677 examples [00:05, 905.14 examples/s]4768 examples [00:05, 906.00 examples/s]4859 examples [00:05, 906.60 examples/s]4950 examples [00:05, 905.23 examples/s]5041 examples [00:05, 903.94 examples/s]5132 examples [00:05, 898.55 examples/s]5222 examples [00:05, 898.96 examples/s]5312 examples [00:05, 896.49 examples/s]5404 examples [00:06, 902.64 examples/s]5496 examples [00:06, 907.74 examples/s]5589 examples [00:06, 911.41 examples/s]5683 examples [00:06, 919.31 examples/s]5778 examples [00:06, 925.94 examples/s]5871 examples [00:06, 911.45 examples/s]5967 examples [00:06, 924.32 examples/s]6060 examples [00:06, 913.23 examples/s]6152 examples [00:06, 914.30 examples/s]6245 examples [00:07, 917.09 examples/s]6338 examples [00:07, 920.08 examples/s]6431 examples [00:07, 923.01 examples/s]6525 examples [00:07, 927.65 examples/s]6618 examples [00:07, 927.50 examples/s]6711 examples [00:07, 915.27 examples/s]6803 examples [00:07, 901.52 examples/s]6894 examples [00:07, 864.66 examples/s]6981 examples [00:07, 819.16 examples/s]7073 examples [00:07, 846.76 examples/s]7166 examples [00:08, 869.73 examples/s]7254 examples [00:08, 864.49 examples/s]7349 examples [00:08, 888.06 examples/s]7439 examples [00:08, 886.27 examples/s]7528 examples [00:08, 882.15 examples/s]7620 examples [00:08, 892.30 examples/s]7711 examples [00:08, 894.81 examples/s]7801 examples [00:08, 895.44 examples/s]7892 examples [00:08, 899.38 examples/s]7983 examples [00:08, 894.64 examples/s]8074 examples [00:09, 896.42 examples/s]8164 examples [00:09, 888.28 examples/s]8254 examples [00:09, 891.72 examples/s]8345 examples [00:09, 896.02 examples/s]8435 examples [00:09, 896.68 examples/s]8526 examples [00:09, 899.63 examples/s]8617 examples [00:09, 901.72 examples/s]8708 examples [00:09, 902.49 examples/s]8805 examples [00:09, 921.38 examples/s]8898 examples [00:09, 917.57 examples/s]8990 examples [00:10, 915.00 examples/s]9082 examples [00:10, 905.63 examples/s]9173 examples [00:10, 900.03 examples/s]9264 examples [00:10, 899.79 examples/s]9355 examples [00:10, 901.24 examples/s]9447 examples [00:10, 905.51 examples/s]9538 examples [00:10, 900.85 examples/s]9633 examples [00:10, 913.20 examples/s]9727 examples [00:10, 921.06 examples/s]9820 examples [00:10, 923.00 examples/s]9913 examples [00:11, 921.70 examples/s]10006 examples [00:11, 854.08 examples/s]10098 examples [00:11, 871.82 examples/s]10190 examples [00:11, 884.43 examples/s]10283 examples [00:11, 896.95 examples/s]10374 examples [00:11, 885.58 examples/s]10463 examples [00:11, 885.41 examples/s]10552 examples [00:11, 885.63 examples/s]10643 examples [00:11, 889.94 examples/s]10733 examples [00:12, 890.91 examples/s]10823 examples [00:12, 889.45 examples/s]10914 examples [00:12, 893.51 examples/s]11004 examples [00:12, 894.84 examples/s]11094 examples [00:12, 875.75 examples/s]11184 examples [00:12, 881.65 examples/s]11275 examples [00:12, 889.13 examples/s]11364 examples [00:12, 887.52 examples/s]11453 examples [00:12, 861.49 examples/s]11549 examples [00:12, 888.10 examples/s]11641 examples [00:13, 897.16 examples/s]11731 examples [00:13, 897.68 examples/s]11823 examples [00:13, 903.97 examples/s]11914 examples [00:13, 897.74 examples/s]12004 examples [00:13, 880.65 examples/s]12095 examples [00:13, 887.38 examples/s]12186 examples [00:13, 893.69 examples/s]12276 examples [00:13, 892.06 examples/s]12366 examples [00:13, 889.84 examples/s]12459 examples [00:13, 898.82 examples/s]12550 examples [00:14, 899.83 examples/s]12641 examples [00:14, 901.26 examples/s]12732 examples [00:14, 898.34 examples/s]12822 examples [00:14, 879.15 examples/s]12911 examples [00:14, 877.17 examples/s]13002 examples [00:14, 884.47 examples/s]13092 examples [00:14, 888.71 examples/s]13182 examples [00:14, 891.65 examples/s]13272 examples [00:14, 892.16 examples/s]13362 examples [00:14, 894.48 examples/s]13452 examples [00:15, 893.53 examples/s]13542 examples [00:15, 891.90 examples/s]13632 examples [00:15, 889.44 examples/s]13723 examples [00:15, 892.90 examples/s]13813 examples [00:15, 889.35 examples/s]13902 examples [00:15, 889.06 examples/s]13993 examples [00:15, 892.95 examples/s]14083 examples [00:15, 891.51 examples/s]14173 examples [00:15, 889.49 examples/s]14262 examples [00:15, 887.47 examples/s]14351 examples [00:16, 880.77 examples/s]14440 examples [00:16, 882.49 examples/s]14529 examples [00:16, 876.00 examples/s]14617 examples [00:16, 872.26 examples/s]14705 examples [00:16, 873.78 examples/s]14794 examples [00:16, 877.02 examples/s]14884 examples [00:16, 882.35 examples/s]14973 examples [00:16, 873.22 examples/s]15062 examples [00:16, 876.63 examples/s]15152 examples [00:17, 882.67 examples/s]15242 examples [00:17, 887.23 examples/s]15331 examples [00:17, 887.75 examples/s]15422 examples [00:17, 893.28 examples/s]15512 examples [00:17, 881.43 examples/s]15604 examples [00:17, 892.42 examples/s]15696 examples [00:17, 898.64 examples/s]15786 examples [00:17, 865.88 examples/s]15876 examples [00:17, 873.31 examples/s]15970 examples [00:17, 891.17 examples/s]16060 examples [00:18, 854.02 examples/s]16146 examples [00:18, 832.42 examples/s]16239 examples [00:18, 857.66 examples/s]16330 examples [00:18, 872.18 examples/s]16423 examples [00:18, 887.02 examples/s]16513 examples [00:18, 853.00 examples/s]16606 examples [00:18, 873.64 examples/s]16697 examples [00:18, 881.78 examples/s]16786 examples [00:18, 883.59 examples/s]16879 examples [00:18, 893.91 examples/s]16971 examples [00:19, 899.71 examples/s]17062 examples [00:19, 902.08 examples/s]17153 examples [00:19, 902.66 examples/s]17244 examples [00:19, 899.41 examples/s]17335 examples [00:19, 900.71 examples/s]17430 examples [00:19, 914.27 examples/s]17523 examples [00:19, 916.95 examples/s]17615 examples [00:19, 889.94 examples/s]17705 examples [00:19, 890.83 examples/s]17795 examples [00:19, 889.94 examples/s]17886 examples [00:20, 893.71 examples/s]17976 examples [00:20, 893.32 examples/s]18066 examples [00:20, 889.58 examples/s]18159 examples [00:20, 899.65 examples/s]18250 examples [00:20, 896.66 examples/s]18343 examples [00:20, 906.07 examples/s]18438 examples [00:20, 916.25 examples/s]18531 examples [00:20, 919.76 examples/s]18624 examples [00:20, 922.03 examples/s]18719 examples [00:21, 929.09 examples/s]18812 examples [00:21, 927.65 examples/s]18905 examples [00:21, 926.20 examples/s]18998 examples [00:21, 923.97 examples/s]19091 examples [00:21, 907.43 examples/s]19185 examples [00:21, 916.43 examples/s]19277 examples [00:21, 916.68 examples/s]19369 examples [00:21, 917.12 examples/s]19464 examples [00:21, 923.92 examples/s]19557 examples [00:21, 910.52 examples/s]19649 examples [00:22, 910.14 examples/s]19742 examples [00:22, 913.67 examples/s]19834 examples [00:22, 914.92 examples/s]19928 examples [00:22, 921.92 examples/s]20021 examples [00:22, 865.07 examples/s]20113 examples [00:22, 878.17 examples/s]20206 examples [00:22, 890.72 examples/s]20296 examples [00:22, 883.79 examples/s]20385 examples [00:22, 867.60 examples/s]20473 examples [00:22, 862.70 examples/s]20565 examples [00:23, 878.35 examples/s]20657 examples [00:23, 890.10 examples/s]20747 examples [00:23, 876.66 examples/s]20841 examples [00:23, 892.76 examples/s]20931 examples [00:23, 893.49 examples/s]21021 examples [00:23, 872.62 examples/s]21111 examples [00:23, 878.67 examples/s]21200 examples [00:23, 881.66 examples/s]21290 examples [00:23, 885.09 examples/s]21379 examples [00:23, 885.75 examples/s]21468 examples [00:24, 886.15 examples/s]21558 examples [00:24, 887.62 examples/s]21649 examples [00:24, 891.57 examples/s]21740 examples [00:24, 894.09 examples/s]21830 examples [00:24, 887.99 examples/s]21921 examples [00:24, 892.24 examples/s]22012 examples [00:24, 895.15 examples/s]22103 examples [00:24, 898.93 examples/s]22194 examples [00:24, 901.57 examples/s]22285 examples [00:24, 897.20 examples/s]22376 examples [00:25, 898.52 examples/s]22466 examples [00:25, 893.43 examples/s]22557 examples [00:25, 897.24 examples/s]22647 examples [00:25, 892.94 examples/s]22737 examples [00:25, 882.98 examples/s]22828 examples [00:25, 890.76 examples/s]22920 examples [00:25, 898.87 examples/s]23012 examples [00:25, 902.81 examples/s]23105 examples [00:25, 908.37 examples/s]23196 examples [00:26, 906.93 examples/s]23290 examples [00:26, 913.89 examples/s]23383 examples [00:26, 916.25 examples/s]23475 examples [00:26, 909.37 examples/s]23566 examples [00:26, 902.14 examples/s]23657 examples [00:26, 886.05 examples/s]23746 examples [00:26, 865.82 examples/s]23833 examples [00:26, 865.89 examples/s]23921 examples [00:26, 867.89 examples/s]24011 examples [00:26, 875.18 examples/s]24103 examples [00:27, 886.71 examples/s]24193 examples [00:27, 889.40 examples/s]24285 examples [00:27, 897.10 examples/s]24377 examples [00:27, 903.54 examples/s]24469 examples [00:27, 906.34 examples/s]24560 examples [00:27, 904.47 examples/s]24651 examples [00:27, 902.06 examples/s]24742 examples [00:27, 862.65 examples/s]24829 examples [00:27, 832.65 examples/s]24913 examples [00:27, 821.87 examples/s]24996 examples [00:28, 813.44 examples/s]25087 examples [00:28, 838.80 examples/s]25178 examples [00:28, 858.12 examples/s]25265 examples [00:28, 842.07 examples/s]25355 examples [00:28, 858.37 examples/s]25444 examples [00:28, 867.41 examples/s]25536 examples [00:28, 881.79 examples/s]25628 examples [00:28, 890.74 examples/s]25720 examples [00:28, 896.66 examples/s]25811 examples [00:28, 898.70 examples/s]25901 examples [00:29, 889.89 examples/s]25993 examples [00:29, 897.33 examples/s]26086 examples [00:29, 905.73 examples/s]26177 examples [00:29, 901.45 examples/s]26268 examples [00:29, 892.63 examples/s]26362 examples [00:29, 904.14 examples/s]26454 examples [00:29, 907.12 examples/s]26548 examples [00:29, 915.45 examples/s]26640 examples [00:29, 904.30 examples/s]26731 examples [00:30, 901.67 examples/s]26822 examples [00:30, 891.92 examples/s]26913 examples [00:30, 895.51 examples/s]27003 examples [00:30, 895.04 examples/s]27093 examples [00:30, 894.12 examples/s]27183 examples [00:30, 892.45 examples/s]27273 examples [00:30, 889.31 examples/s]27363 examples [00:30, 890.75 examples/s]27458 examples [00:30, 906.96 examples/s]27553 examples [00:30, 916.73 examples/s]27646 examples [00:31, 917.65 examples/s]27738 examples [00:31, 904.81 examples/s]27830 examples [00:31, 906.88 examples/s]27923 examples [00:31, 913.20 examples/s]28015 examples [00:31, 911.23 examples/s]28107 examples [00:31, 901.01 examples/s]28198 examples [00:31, 893.31 examples/s]28289 examples [00:31, 897.64 examples/s]28382 examples [00:31, 904.89 examples/s]28474 examples [00:31, 907.17 examples/s]28566 examples [00:32, 910.56 examples/s]28658 examples [00:32, 896.84 examples/s]28750 examples [00:32, 901.93 examples/s]28842 examples [00:32, 905.22 examples/s]28933 examples [00:32, 898.24 examples/s]29023 examples [00:32, 886.46 examples/s]29115 examples [00:32, 895.49 examples/s]29209 examples [00:32, 906.46 examples/s]29300 examples [00:32, 883.84 examples/s]29389 examples [00:32, 881.94 examples/s]29479 examples [00:33, 885.75 examples/s]29568 examples [00:33, 854.73 examples/s]29656 examples [00:33, 861.73 examples/s]29747 examples [00:33, 873.09 examples/s]29840 examples [00:33, 887.92 examples/s]29929 examples [00:33, 872.20 examples/s]30017 examples [00:33, 809.02 examples/s]30105 examples [00:33, 828.47 examples/s]30193 examples [00:33, 840.94 examples/s]30285 examples [00:34, 862.47 examples/s]30376 examples [00:34, 875.30 examples/s]30471 examples [00:34, 894.38 examples/s]30563 examples [00:34, 899.48 examples/s]30654 examples [00:34, 882.54 examples/s]30745 examples [00:34, 889.23 examples/s]30835 examples [00:34, 878.45 examples/s]30924 examples [00:34, 877.03 examples/s]31015 examples [00:34, 885.37 examples/s]31107 examples [00:34, 894.13 examples/s]31201 examples [00:35, 904.69 examples/s]31292 examples [00:35, 906.03 examples/s]31383 examples [00:35, 884.72 examples/s]31472 examples [00:35, 883.78 examples/s]31561 examples [00:35, 884.38 examples/s]31651 examples [00:35, 887.49 examples/s]31742 examples [00:35, 893.13 examples/s]31833 examples [00:35, 896.47 examples/s]31924 examples [00:35, 899.23 examples/s]32016 examples [00:35, 903.38 examples/s]32108 examples [00:36, 907.99 examples/s]32200 examples [00:36, 911.33 examples/s]32292 examples [00:36, 906.72 examples/s]32385 examples [00:36, 912.65 examples/s]32477 examples [00:36, 912.26 examples/s]32569 examples [00:36, 912.40 examples/s]32661 examples [00:36, 891.95 examples/s]32754 examples [00:36, 902.46 examples/s]32845 examples [00:36, 902.47 examples/s]32936 examples [00:36, 896.00 examples/s]33026 examples [00:37, 890.78 examples/s]33116 examples [00:37, 890.59 examples/s]33206 examples [00:37, 872.27 examples/s]33294 examples [00:37, 874.27 examples/s]33385 examples [00:37, 884.05 examples/s]33477 examples [00:37, 892.74 examples/s]33569 examples [00:37, 899.69 examples/s]33660 examples [00:37, 900.08 examples/s]33751 examples [00:37, 884.12 examples/s]33841 examples [00:37, 888.16 examples/s]33933 examples [00:38, 895.61 examples/s]34023 examples [00:38, 896.19 examples/s]34113 examples [00:38, 889.55 examples/s]34202 examples [00:38, 868.70 examples/s]34297 examples [00:38, 889.21 examples/s]34391 examples [00:38, 903.33 examples/s]34484 examples [00:38, 910.16 examples/s]34576 examples [00:38, 889.90 examples/s]34668 examples [00:38, 897.46 examples/s]34760 examples [00:39, 901.99 examples/s]34853 examples [00:39, 908.00 examples/s]34945 examples [00:39, 909.47 examples/s]35037 examples [00:39, 899.39 examples/s]35128 examples [00:39, 901.14 examples/s]35220 examples [00:39, 905.67 examples/s]35312 examples [00:39, 907.07 examples/s]35403 examples [00:39, 907.36 examples/s]35494 examples [00:39, 904.35 examples/s]35585 examples [00:39, 882.69 examples/s]35677 examples [00:40, 892.77 examples/s]35769 examples [00:40, 898.09 examples/s]35861 examples [00:40, 902.54 examples/s]35954 examples [00:40, 908.32 examples/s]36045 examples [00:40, 908.77 examples/s]36136 examples [00:40, 908.07 examples/s]36229 examples [00:40, 914.31 examples/s]36322 examples [00:40, 917.39 examples/s]36416 examples [00:40, 922.98 examples/s]36513 examples [00:40, 933.88 examples/s]36607 examples [00:41, 932.92 examples/s]36701 examples [00:41, 933.40 examples/s]36795 examples [00:41, 928.20 examples/s]36888 examples [00:41, 923.78 examples/s]36981 examples [00:41, 916.39 examples/s]37073 examples [00:41, 914.64 examples/s]37165 examples [00:41, 914.01 examples/s]37257 examples [00:41, 911.51 examples/s]37351 examples [00:41, 917.21 examples/s]37443 examples [00:41, 911.93 examples/s]37535 examples [00:42, 909.83 examples/s]37628 examples [00:42, 913.99 examples/s]37720 examples [00:42, 915.08 examples/s]37812 examples [00:42, 909.75 examples/s]37903 examples [00:42, 905.07 examples/s]37994 examples [00:42, 899.16 examples/s]38084 examples [00:42, 867.03 examples/s]38171 examples [00:42, 864.76 examples/s]38258 examples [00:42, 858.01 examples/s]38350 examples [00:42, 874.81 examples/s]38441 examples [00:43, 883.92 examples/s]38530 examples [00:43, 879.39 examples/s]38624 examples [00:43, 893.74 examples/s]38714 examples [00:43, 893.62 examples/s]38804 examples [00:43, 891.59 examples/s]38894 examples [00:43, 891.70 examples/s]38985 examples [00:43, 896.52 examples/s]39076 examples [00:43, 899.07 examples/s]39166 examples [00:43, 896.70 examples/s]39256 examples [00:43, 866.59 examples/s]39343 examples [00:44, 832.07 examples/s]39432 examples [00:44, 846.15 examples/s]39517 examples [00:44, 843.47 examples/s]39610 examples [00:44, 867.69 examples/s]39698 examples [00:44, 869.54 examples/s]39787 examples [00:44, 873.77 examples/s]39877 examples [00:44, 879.34 examples/s]39966 examples [00:44, 868.88 examples/s]40054 examples [00:44, 806.74 examples/s]40144 examples [00:45, 830.84 examples/s]40235 examples [00:45, 850.81 examples/s]40325 examples [00:45, 864.54 examples/s]40414 examples [00:45, 871.74 examples/s]40506 examples [00:45, 882.99 examples/s]40595 examples [00:45, 883.47 examples/s]40687 examples [00:45, 891.63 examples/s]40779 examples [00:45, 898.46 examples/s]40869 examples [00:45, 895.05 examples/s]40960 examples [00:45, 898.05 examples/s]41050 examples [00:46, 883.14 examples/s]41143 examples [00:46, 896.12 examples/s]41233 examples [00:46, 895.39 examples/s]41326 examples [00:46, 902.86 examples/s]41417 examples [00:46, 904.62 examples/s]41508 examples [00:46, 905.68 examples/s]41602 examples [00:46, 913.07 examples/s]41694 examples [00:46, 874.71 examples/s]41782 examples [00:46, 855.37 examples/s]41871 examples [00:46, 865.26 examples/s]41960 examples [00:47, 872.30 examples/s]42048 examples [00:47, 871.18 examples/s]42138 examples [00:47, 878.90 examples/s]42227 examples [00:47, 870.09 examples/s]42315 examples [00:47, 871.71 examples/s]42405 examples [00:47, 877.25 examples/s]42496 examples [00:47, 886.20 examples/s]42585 examples [00:47, 879.82 examples/s]42674 examples [00:47, 874.12 examples/s]42765 examples [00:48, 882.76 examples/s]42858 examples [00:48, 895.38 examples/s]42951 examples [00:48, 902.93 examples/s]43042 examples [00:48, 900.23 examples/s]43134 examples [00:48, 903.23 examples/s]43227 examples [00:48, 910.05 examples/s]43319 examples [00:48, 903.12 examples/s]43410 examples [00:48, 878.82 examples/s]43500 examples [00:48, 883.41 examples/s]43589 examples [00:48, 879.35 examples/s]43680 examples [00:49, 887.59 examples/s]43775 examples [00:49, 904.08 examples/s]43866 examples [00:49, 894.90 examples/s]43956 examples [00:49, 890.04 examples/s]44046 examples [00:49, 892.77 examples/s]44138 examples [00:49, 897.94 examples/s]44228 examples [00:49, 893.97 examples/s]44322 examples [00:49, 907.29 examples/s]44415 examples [00:49, 912.64 examples/s]44507 examples [00:49, 891.68 examples/s]44597 examples [00:50, 879.83 examples/s]44689 examples [00:50, 889.65 examples/s]44779 examples [00:50, 886.93 examples/s]44868 examples [00:50, 886.68 examples/s]44957 examples [00:50, 884.24 examples/s]45049 examples [00:50, 891.50 examples/s]45141 examples [00:50, 897.26 examples/s]45233 examples [00:50, 902.28 examples/s]45325 examples [00:50, 906.59 examples/s]45416 examples [00:50, 880.97 examples/s]45506 examples [00:51, 886.21 examples/s]45595 examples [00:51, 886.12 examples/s]45686 examples [00:51, 892.05 examples/s]45776 examples [00:51, 892.87 examples/s]45866 examples [00:51, 887.45 examples/s]45955 examples [00:51, 884.01 examples/s]46044 examples [00:51, 870.55 examples/s]46135 examples [00:51, 880.20 examples/s]46227 examples [00:51, 889.40 examples/s]46317 examples [00:51, 870.32 examples/s]46405 examples [00:52, 861.87 examples/s]46496 examples [00:52, 874.17 examples/s]46589 examples [00:52, 887.36 examples/s]46678 examples [00:52, 877.60 examples/s]46769 examples [00:52, 885.18 examples/s]46860 examples [00:52, 892.01 examples/s]46950 examples [00:52, 889.31 examples/s]47039 examples [00:52, 868.76 examples/s]47127 examples [00:52, 871.72 examples/s]47222 examples [00:53, 891.62 examples/s]47312 examples [00:53, 887.07 examples/s]47401 examples [00:53, 864.08 examples/s]47493 examples [00:53, 879.77 examples/s]47582 examples [00:53, 882.24 examples/s]47671 examples [00:53, 879.42 examples/s]47760 examples [00:53, 882.02 examples/s]47849 examples [00:53, 875.72 examples/s]47941 examples [00:53, 886.59 examples/s]48030 examples [00:53, 873.29 examples/s]48118 examples [00:54, 875.16 examples/s]48209 examples [00:54, 882.59 examples/s]48298 examples [00:54, 863.54 examples/s]48388 examples [00:54, 872.59 examples/s]48476 examples [00:54, 866.65 examples/s]48568 examples [00:54, 881.55 examples/s]48657 examples [00:54, 873.46 examples/s]48745 examples [00:54, 872.44 examples/s]48835 examples [00:54, 880.45 examples/s]48924 examples [00:54, 877.64 examples/s]49014 examples [00:55, 884.00 examples/s]49109 examples [00:55, 901.91 examples/s]49200 examples [00:55, 885.83 examples/s]49289 examples [00:55, 885.71 examples/s]49379 examples [00:55, 888.88 examples/s]49471 examples [00:55, 895.27 examples/s]49562 examples [00:55, 899.04 examples/s]49654 examples [00:55, 903.55 examples/s]49747 examples [00:55, 910.50 examples/s]49839 examples [00:55, 906.59 examples/s]49930 examples [00:56, 906.65 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█▏        | 5650/50000 [00:00<00:00, 56499.65 examples/s] 34%|███▍      | 17081/50000 [00:00<00:00, 66604.76 examples/s] 57%|█████▋    | 28316/50000 [00:00<00:00, 75872.45 examples/s] 80%|███████▉  | 39849/50000 [00:00<00:00, 84548.59 examples/s]                                                               0 examples [00:00, ? examples/s]68 examples [00:00, 675.10 examples/s]158 examples [00:00, 728.81 examples/s]245 examples [00:00, 766.05 examples/s]334 examples [00:00, 798.99 examples/s]427 examples [00:00, 832.13 examples/s]521 examples [00:00, 861.53 examples/s]615 examples [00:00, 883.64 examples/s]705 examples [00:00, 886.20 examples/s]797 examples [00:00, 895.90 examples/s]891 examples [00:01, 906.61 examples/s]986 examples [00:01, 917.85 examples/s]1077 examples [00:01, 893.92 examples/s]1169 examples [00:01, 899.41 examples/s]1259 examples [00:01, 895.04 examples/s]1350 examples [00:01, 897.31 examples/s]1443 examples [00:01, 905.04 examples/s]1534 examples [00:01, 903.28 examples/s]1627 examples [00:01, 909.26 examples/s]1722 examples [00:01, 918.51 examples/s]1817 examples [00:02, 927.66 examples/s]1910 examples [00:02, 898.98 examples/s]2008 examples [00:02, 920.29 examples/s]2103 examples [00:02, 928.09 examples/s]2197 examples [00:02, 931.40 examples/s]2291 examples [00:02, 928.93 examples/s]2385 examples [00:02, 930.53 examples/s]2479 examples [00:02, 929.56 examples/s]2573 examples [00:02, 928.21 examples/s]2666 examples [00:02, 890.65 examples/s]2756 examples [00:03, 893.06 examples/s]2854 examples [00:03, 916.34 examples/s]2948 examples [00:03, 922.45 examples/s]3041 examples [00:03, 894.80 examples/s]3137 examples [00:03, 913.26 examples/s]3229 examples [00:03, 915.14 examples/s]3322 examples [00:03, 917.72 examples/s]3414 examples [00:03, 913.62 examples/s]3506 examples [00:03, 912.32 examples/s]3600 examples [00:03, 920.12 examples/s]3693 examples [00:04, 918.37 examples/s]3786 examples [00:04, 919.86 examples/s]3879 examples [00:04, 918.99 examples/s]3973 examples [00:04, 923.71 examples/s]4067 examples [00:04, 926.19 examples/s]4160 examples [00:04, 887.36 examples/s]4250 examples [00:04, 875.29 examples/s]4338 examples [00:04, 856.01 examples/s]4424 examples [00:04, 840.85 examples/s]4518 examples [00:05, 868.18 examples/s]4607 examples [00:05, 873.69 examples/s]4699 examples [00:05, 886.85 examples/s]4792 examples [00:05, 897.56 examples/s]4884 examples [00:05, 903.18 examples/s]4977 examples [00:05, 909.21 examples/s]5071 examples [00:05, 915.60 examples/s]5163 examples [00:05, 914.92 examples/s]5255 examples [00:05, 899.85 examples/s]5346 examples [00:05, 901.92 examples/s]5438 examples [00:06, 904.75 examples/s]5531 examples [00:06, 910.69 examples/s]5623 examples [00:06, 888.90 examples/s]5716 examples [00:06, 898.35 examples/s]5806 examples [00:06, 884.07 examples/s]5900 examples [00:06, 897.61 examples/s]5990 examples [00:06, 880.76 examples/s]6079 examples [00:06, 851.18 examples/s]6169 examples [00:06, 862.17 examples/s]6262 examples [00:06, 881.41 examples/s]6355 examples [00:07, 894.00 examples/s]6445 examples [00:07, 893.82 examples/s]6544 examples [00:07, 916.02 examples/s]6636 examples [00:07, 916.96 examples/s]6728 examples [00:07, 915.84 examples/s]6822 examples [00:07, 922.16 examples/s]6915 examples [00:07, 923.37 examples/s]7008 examples [00:07, 920.72 examples/s]7102 examples [00:07, 923.61 examples/s]7196 examples [00:07, 927.88 examples/s]7290 examples [00:08, 930.26 examples/s]7384 examples [00:08, 891.51 examples/s]7483 examples [00:08, 916.62 examples/s]7577 examples [00:08, 921.33 examples/s]7673 examples [00:08, 931.95 examples/s]7769 examples [00:08, 940.04 examples/s]7865 examples [00:08, 945.24 examples/s]7960 examples [00:08, 944.18 examples/s]8055 examples [00:08, 917.67 examples/s]8147 examples [00:09, 900.58 examples/s]8239 examples [00:09, 905.72 examples/s]8332 examples [00:09, 911.80 examples/s]8426 examples [00:09, 919.70 examples/s]8519 examples [00:09, 922.26 examples/s]8614 examples [00:09, 930.29 examples/s]8708 examples [00:09, 929.90 examples/s]8802 examples [00:09, 922.31 examples/s]8895 examples [00:09, 920.83 examples/s]8989 examples [00:09, 924.10 examples/s]9084 examples [00:10, 930.10 examples/s]9178 examples [00:10, 919.60 examples/s]9272 examples [00:10, 924.00 examples/s]9365 examples [00:10, 921.17 examples/s]9458 examples [00:10, 896.01 examples/s]9548 examples [00:10, 877.48 examples/s]9640 examples [00:10, 888.81 examples/s]9731 examples [00:10, 894.21 examples/s]9821 examples [00:10, 881.11 examples/s]9910 examples [00:10, 877.99 examples/s]10000 examples [00:11, 883.30 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s] 98%|█████████▊| 9789/10000 [00:00<00:00, 97888.23 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1JB9O2/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1JB9O2/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f012e7d7158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f012e7d7158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f012e7d7158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f017a1b40b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f00ecaa1438>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f012e7d7158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f012e7d7158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f012e7d7158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f012ccbd400>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f012ccbd358>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 12.65 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.21 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.45 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.45 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.86 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.86 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.28 file/s]2020-08-07 18:08:55.384161: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-07 18:08:55.388565: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-07 18:08:55.388769: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555dbe80a7e0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-07 18:08:55.388788: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147646.65it/s] 62%|██████▏   | 6103040/9912422 [00:00<00:18, 210704.67it/s]9920512it [00:00, 39528769.67it/s]                           
0it [00:00, ?it/s]32768it [00:00, 420303.90it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 159812.74it/s]1654784it [00:00, 11141862.33it/s]                         
0it [00:00, ?it/s]8192it [00:00, 166697.74it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

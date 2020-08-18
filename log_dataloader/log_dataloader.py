
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f8ee66df598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f8ee66df598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8f51391400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8f51391400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f8f705adea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f8f705adea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8efe6cc0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8efe6cc0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8efe6cc0d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:20, 123348.38it/s] 70%|███████   | 6979584/9912422 [00:00<00:16, 176078.15it/s]9920512it [00:00, 37763574.19it/s]                           
0it [00:00, ?it/s]32768it [00:00, 654103.66it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 474326.51it/s]1654784it [00:00, 11542039.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 219819.32it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8efcbc6780>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8efc868a20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f8efe6c4d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f8efe6c4d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f8efe6c4d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:27,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:27,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:26,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:26,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:25,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:25,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:24,  1.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:59,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:59,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:58,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:58,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:57,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:57,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:57,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:56,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:56,  2.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:39,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:39,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:39,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:38,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:38,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:38,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:38,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:37,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:37,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:37,  3.65 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:26,  5.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:26,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:26,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:25,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:25,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:24,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:24,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:24,  5.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:17,  7.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:17,  7.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:17,  7.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:17,  7.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:17,  7.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:16,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:16,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:16,  7.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:11,  9.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
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

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10,  9.90 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.40 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:05, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 17.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 23.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 30.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 30.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 37.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 37.66 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 45.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 53.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 53.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 67.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 71.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 75.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.28 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.39s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.28 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 34.39 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.71s/ url]
0 examples [00:00, ? examples/s]2020-08-18 06:07:23.749467: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-18 06:07:23.762383: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-18 06:07:23.762695: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55fe6c20ba50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-18 06:07:23.762716: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
12 examples [00:00, 118.88 examples/s]105 examples [00:00, 161.00 examples/s]214 examples [00:00, 216.24 examples/s]310 examples [00:00, 281.48 examples/s]411 examples [00:00, 358.82 examples/s]518 examples [00:00, 448.18 examples/s]619 examples [00:00, 537.62 examples/s]719 examples [00:00, 623.77 examples/s]818 examples [00:00, 700.21 examples/s]913 examples [00:01, 758.89 examples/s]1008 examples [00:01, 805.81 examples/s]1107 examples [00:01, 853.28 examples/s]1208 examples [00:01, 893.12 examples/s]1306 examples [00:01, 912.64 examples/s]1403 examples [00:01, 926.93 examples/s]1500 examples [00:01, 917.38 examples/s]1595 examples [00:01, 926.64 examples/s]1692 examples [00:01, 937.11 examples/s]1788 examples [00:01, 938.52 examples/s]1883 examples [00:02, 928.46 examples/s]1977 examples [00:02, 931.00 examples/s]2071 examples [00:02, 918.49 examples/s]2175 examples [00:02, 945.99 examples/s]2271 examples [00:02, 943.03 examples/s]2368 examples [00:02, 950.43 examples/s]2464 examples [00:02, 949.84 examples/s]2562 examples [00:02, 956.94 examples/s]2670 examples [00:02, 990.60 examples/s]2770 examples [00:02, 982.73 examples/s]2869 examples [00:03, 970.78 examples/s]2967 examples [00:03, 965.79 examples/s]3064 examples [00:03, 961.15 examples/s]3161 examples [00:03, 947.44 examples/s]3256 examples [00:03, 921.08 examples/s]3349 examples [00:03, 921.38 examples/s]3442 examples [00:03, 908.05 examples/s]3535 examples [00:03, 913.61 examples/s]3627 examples [00:03, 914.14 examples/s]3722 examples [00:03, 923.26 examples/s]3825 examples [00:04, 950.78 examples/s]3924 examples [00:04, 960.31 examples/s]4031 examples [00:04, 988.07 examples/s]4131 examples [00:04, 989.27 examples/s]4231 examples [00:04, 979.46 examples/s]4336 examples [00:04, 997.12 examples/s]4436 examples [00:04, 965.10 examples/s]4534 examples [00:04, 968.23 examples/s]4632 examples [00:04, 970.13 examples/s]4735 examples [00:04, 987.32 examples/s]4837 examples [00:05, 991.98 examples/s]4938 examples [00:05, 996.78 examples/s]5039 examples [00:05, 997.67 examples/s]5147 examples [00:05, 1019.62 examples/s]5250 examples [00:05, 1022.06 examples/s]5353 examples [00:05, 994.59 examples/s] 5457 examples [00:05, 1007.33 examples/s]5560 examples [00:05, 1013.06 examples/s]5669 examples [00:05, 1031.66 examples/s]5775 examples [00:06, 1037.99 examples/s]5879 examples [00:06, 1019.15 examples/s]5982 examples [00:06, 1014.23 examples/s]6084 examples [00:06, 997.58 examples/s] 6184 examples [00:06, 982.49 examples/s]6283 examples [00:06, 957.71 examples/s]6388 examples [00:06, 981.64 examples/s]6487 examples [00:06, 975.98 examples/s]6586 examples [00:06, 978.12 examples/s]6684 examples [00:06, 976.33 examples/s]6782 examples [00:07, 969.75 examples/s]6880 examples [00:07, 947.30 examples/s]6981 examples [00:07, 963.42 examples/s]7083 examples [00:07, 978.53 examples/s]7184 examples [00:07, 986.92 examples/s]7286 examples [00:07, 994.08 examples/s]7386 examples [00:07, 989.88 examples/s]7486 examples [00:07, 991.12 examples/s]7586 examples [00:07, 960.13 examples/s]7690 examples [00:07, 980.04 examples/s]7790 examples [00:08, 985.36 examples/s]7889 examples [00:08, 977.58 examples/s]7987 examples [00:08, 973.46 examples/s]8091 examples [00:08, 989.95 examples/s]8197 examples [00:08, 1007.97 examples/s]8298 examples [00:08, 971.71 examples/s] 8396 examples [00:08, 955.93 examples/s]8492 examples [00:08, 945.66 examples/s]8587 examples [00:08, 940.95 examples/s]8684 examples [00:09, 949.12 examples/s]8784 examples [00:09, 963.64 examples/s]8881 examples [00:09, 958.63 examples/s]8980 examples [00:09, 965.79 examples/s]9077 examples [00:09, 951.19 examples/s]9179 examples [00:09, 968.89 examples/s]9279 examples [00:09, 975.45 examples/s]9377 examples [00:09, 962.50 examples/s]9480 examples [00:09, 981.73 examples/s]9581 examples [00:09, 989.14 examples/s]9682 examples [00:10, 994.90 examples/s]9792 examples [00:10, 1022.30 examples/s]9895 examples [00:10, 1006.24 examples/s]9998 examples [00:10, 1008.10 examples/s]10099 examples [00:10, 941.79 examples/s]10199 examples [00:10, 955.77 examples/s]10301 examples [00:10, 973.97 examples/s]10400 examples [00:10, 973.38 examples/s]10499 examples [00:10, 977.87 examples/s]10598 examples [00:10, 970.19 examples/s]10700 examples [00:11, 983.31 examples/s]10809 examples [00:11, 1012.35 examples/s]10911 examples [00:11, 1000.97 examples/s]11013 examples [00:11, 1005.19 examples/s]11120 examples [00:11, 1023.03 examples/s]11223 examples [00:11, 1006.74 examples/s]11324 examples [00:11, 1003.08 examples/s]11425 examples [00:11, 990.42 examples/s] 11527 examples [00:11, 995.87 examples/s]11627 examples [00:11, 975.09 examples/s]11726 examples [00:12, 977.19 examples/s]11824 examples [00:12, 962.49 examples/s]11921 examples [00:12, 952.59 examples/s]12025 examples [00:12, 976.15 examples/s]12124 examples [00:12, 979.40 examples/s]12223 examples [00:12, 974.89 examples/s]12327 examples [00:12, 992.82 examples/s]12427 examples [00:12, 988.98 examples/s]12529 examples [00:12, 996.67 examples/s]12629 examples [00:13, 986.09 examples/s]12735 examples [00:13, 1006.63 examples/s]12836 examples [00:13, 988.34 examples/s] 12936 examples [00:13, 974.09 examples/s]13037 examples [00:13, 982.53 examples/s]13139 examples [00:13, 991.00 examples/s]13239 examples [00:13, 993.39 examples/s]13339 examples [00:13, 952.62 examples/s]13435 examples [00:13, 948.96 examples/s]13531 examples [00:13, 951.60 examples/s]13630 examples [00:14, 962.58 examples/s]13732 examples [00:14, 976.60 examples/s]13830 examples [00:14, 967.36 examples/s]13927 examples [00:14, 953.77 examples/s]14024 examples [00:14, 955.76 examples/s]14120 examples [00:14, 955.38 examples/s]14220 examples [00:14, 965.45 examples/s]14318 examples [00:14, 967.79 examples/s]14415 examples [00:14, 967.19 examples/s]14519 examples [00:14, 986.12 examples/s]14618 examples [00:15, 986.83 examples/s]14721 examples [00:15, 998.87 examples/s]14821 examples [00:15, 996.38 examples/s]14921 examples [00:15, 966.42 examples/s]15018 examples [00:15, 966.07 examples/s]15115 examples [00:15, 961.71 examples/s]15219 examples [00:15, 982.03 examples/s]15318 examples [00:15, 978.80 examples/s]15417 examples [00:15, 978.29 examples/s]15525 examples [00:15, 1005.84 examples/s]15635 examples [00:16, 1030.49 examples/s]15739 examples [00:16, 1027.94 examples/s]15843 examples [00:16, 1016.28 examples/s]15945 examples [00:16, 1003.17 examples/s]16053 examples [00:16, 1022.92 examples/s]16156 examples [00:16, 1022.69 examples/s]16259 examples [00:16, 1002.76 examples/s]16360 examples [00:16, 989.78 examples/s] 16460 examples [00:16, 975.49 examples/s]16558 examples [00:17, 962.86 examples/s]16655 examples [00:17, 951.80 examples/s]16753 examples [00:17, 958.89 examples/s]16850 examples [00:17, 961.12 examples/s]16947 examples [00:17, 961.02 examples/s]17047 examples [00:17, 971.61 examples/s]17150 examples [00:17, 985.67 examples/s]17249 examples [00:17, 984.53 examples/s]17353 examples [00:17, 999.82 examples/s]17454 examples [00:17, 977.42 examples/s]17552 examples [00:18, 975.78 examples/s]17652 examples [00:18, 980.98 examples/s]17759 examples [00:18, 1003.97 examples/s]17860 examples [00:18, 963.12 examples/s] 17962 examples [00:18, 978.31 examples/s]18068 examples [00:18, 1001.33 examples/s]18169 examples [00:18, 1000.71 examples/s]18270 examples [00:18, 995.59 examples/s] 18370 examples [00:18, 991.45 examples/s]18470 examples [00:18, 962.15 examples/s]18570 examples [00:19, 970.79 examples/s]18669 examples [00:19, 974.52 examples/s]18767 examples [00:19, 966.94 examples/s]18864 examples [00:19, 949.84 examples/s]18960 examples [00:19, 935.87 examples/s]19056 examples [00:19, 941.78 examples/s]19160 examples [00:19, 968.38 examples/s]19265 examples [00:19, 989.46 examples/s]19367 examples [00:19, 997.00 examples/s]19467 examples [00:19, 987.24 examples/s]19569 examples [00:20, 994.37 examples/s]19669 examples [00:20, 993.22 examples/s]19775 examples [00:20, 1012.18 examples/s]19877 examples [00:20, 997.58 examples/s] 19977 examples [00:20, 986.62 examples/s]20076 examples [00:20, 917.99 examples/s]20173 examples [00:20, 932.50 examples/s]20273 examples [00:20, 950.27 examples/s]20373 examples [00:20, 964.66 examples/s]20478 examples [00:21, 988.14 examples/s]20578 examples [00:21, 979.98 examples/s]20677 examples [00:21, 963.87 examples/s]20774 examples [00:21, 950.80 examples/s]20874 examples [00:21, 962.88 examples/s]20971 examples [00:21, 964.06 examples/s]21070 examples [00:21, 970.28 examples/s]21168 examples [00:21, 970.90 examples/s]21268 examples [00:21, 970.13 examples/s]21366 examples [00:21, 953.60 examples/s]21467 examples [00:22, 967.05 examples/s]21564 examples [00:22, 965.23 examples/s]21661 examples [00:22, 959.61 examples/s]21760 examples [00:22, 966.08 examples/s]21859 examples [00:22, 970.98 examples/s]21962 examples [00:22, 985.72 examples/s]22061 examples [00:22, 974.34 examples/s]22159 examples [00:22, 970.28 examples/s]22261 examples [00:22, 984.38 examples/s]22360 examples [00:22, 981.48 examples/s]22461 examples [00:23, 989.53 examples/s]22562 examples [00:23, 994.48 examples/s]22662 examples [00:23, 989.27 examples/s]22761 examples [00:23, 983.33 examples/s]22871 examples [00:23, 1015.35 examples/s]22973 examples [00:23, 1015.03 examples/s]23075 examples [00:23, 981.60 examples/s] 23183 examples [00:23, 1007.84 examples/s]23285 examples [00:23, 1010.81 examples/s]23387 examples [00:24, 962.11 examples/s] 23484 examples [00:24, 951.08 examples/s]23580 examples [00:24, 938.25 examples/s]23680 examples [00:24, 955.86 examples/s]23782 examples [00:24, 971.44 examples/s]23880 examples [00:24, 956.54 examples/s]23980 examples [00:24, 967.17 examples/s]24077 examples [00:24, 963.95 examples/s]24184 examples [00:24, 993.03 examples/s]24284 examples [00:24, 992.57 examples/s]24384 examples [00:25, 969.27 examples/s]24482 examples [00:25, 952.78 examples/s]24579 examples [00:25, 957.48 examples/s]24677 examples [00:25, 962.84 examples/s]24777 examples [00:25, 972.24 examples/s]24881 examples [00:25, 989.52 examples/s]24981 examples [00:25, 990.55 examples/s]25084 examples [00:25, 999.88 examples/s]25185 examples [00:25, 991.51 examples/s]25285 examples [00:25, 984.89 examples/s]25384 examples [00:26, 976.10 examples/s]25487 examples [00:26, 990.52 examples/s]25588 examples [00:26, 994.33 examples/s]25688 examples [00:26, 979.82 examples/s]25787 examples [00:26, 959.96 examples/s]25884 examples [00:26, 952.90 examples/s]25981 examples [00:26, 955.29 examples/s]26084 examples [00:26, 974.35 examples/s]26182 examples [00:26, 938.13 examples/s]26277 examples [00:26, 937.87 examples/s]26373 examples [00:27, 943.91 examples/s]26468 examples [00:27, 930.41 examples/s]26568 examples [00:27, 947.77 examples/s]26663 examples [00:27, 931.48 examples/s]26757 examples [00:27, 926.20 examples/s]26851 examples [00:27, 928.13 examples/s]26947 examples [00:27, 935.01 examples/s]27042 examples [00:27, 939.37 examples/s]27137 examples [00:27, 930.38 examples/s]27231 examples [00:28, 929.28 examples/s]27324 examples [00:28, 923.73 examples/s]27425 examples [00:28, 946.27 examples/s]27521 examples [00:28, 949.81 examples/s]27618 examples [00:28, 955.59 examples/s]27714 examples [00:28, 924.84 examples/s]27810 examples [00:28, 934.88 examples/s]27909 examples [00:28, 950.02 examples/s]28010 examples [00:28, 965.87 examples/s]28107 examples [00:28, 960.16 examples/s]28204 examples [00:29, 906.24 examples/s]28302 examples [00:29, 926.82 examples/s]28397 examples [00:29, 931.80 examples/s]28492 examples [00:29, 935.95 examples/s]28587 examples [00:29, 939.44 examples/s]28683 examples [00:29, 945.23 examples/s]28779 examples [00:29, 948.15 examples/s]28875 examples [00:29, 949.69 examples/s]28979 examples [00:29, 973.10 examples/s]29077 examples [00:29, 962.19 examples/s]29174 examples [00:30, 955.11 examples/s]29282 examples [00:30, 988.77 examples/s]29386 examples [00:30, 1001.26 examples/s]29487 examples [00:30, 996.40 examples/s] 29587 examples [00:30, 990.45 examples/s]29687 examples [00:30, 971.66 examples/s]29785 examples [00:30, 967.74 examples/s]29882 examples [00:30, 960.09 examples/s]29980 examples [00:30, 963.88 examples/s]30077 examples [00:31, 896.33 examples/s]30179 examples [00:31, 928.19 examples/s]30280 examples [00:31, 949.81 examples/s]30379 examples [00:31, 960.57 examples/s]30476 examples [00:31, 962.50 examples/s]30573 examples [00:31, 960.38 examples/s]30676 examples [00:31, 978.06 examples/s]30778 examples [00:31, 988.53 examples/s]30879 examples [00:31, 994.28 examples/s]30979 examples [00:31, 989.42 examples/s]31079 examples [00:32, 970.01 examples/s]31180 examples [00:32, 981.66 examples/s]31279 examples [00:32, 956.92 examples/s]31375 examples [00:32, 948.80 examples/s]31477 examples [00:32, 967.02 examples/s]31574 examples [00:32, 963.62 examples/s]31671 examples [00:32, 959.81 examples/s]31769 examples [00:32, 965.21 examples/s]31866 examples [00:32, 943.68 examples/s]31963 examples [00:32, 950.17 examples/s]32060 examples [00:33, 954.56 examples/s]32160 examples [00:33, 965.66 examples/s]32258 examples [00:33, 968.91 examples/s]32355 examples [00:33, 961.55 examples/s]32454 examples [00:33, 969.62 examples/s]32556 examples [00:33, 982.21 examples/s]32655 examples [00:33, 970.67 examples/s]32753 examples [00:33, 964.77 examples/s]32852 examples [00:33, 971.92 examples/s]32950 examples [00:33, 966.76 examples/s]33047 examples [00:34, 947.02 examples/s]33149 examples [00:34, 967.75 examples/s]33251 examples [00:34, 981.20 examples/s]33350 examples [00:34, 963.84 examples/s]33449 examples [00:34, 968.45 examples/s]33546 examples [00:34, 962.87 examples/s]33643 examples [00:34, 951.88 examples/s]33748 examples [00:34, 977.25 examples/s]33851 examples [00:34, 990.44 examples/s]33952 examples [00:34, 994.89 examples/s]34052 examples [00:35, 965.87 examples/s]34153 examples [00:35, 977.05 examples/s]34255 examples [00:35, 987.53 examples/s]34354 examples [00:35, 977.51 examples/s]34452 examples [00:35, 952.45 examples/s]34548 examples [00:35, 929.16 examples/s]34642 examples [00:35, 922.74 examples/s]34740 examples [00:35, 937.66 examples/s]34834 examples [00:35, 933.99 examples/s]34936 examples [00:36, 957.63 examples/s]35033 examples [00:36, 952.81 examples/s]35132 examples [00:36, 963.15 examples/s]35233 examples [00:36, 974.80 examples/s]35331 examples [00:36, 967.29 examples/s]35428 examples [00:36, 959.06 examples/s]35525 examples [00:36, 949.96 examples/s]35621 examples [00:36, 950.10 examples/s]35719 examples [00:36, 957.39 examples/s]35816 examples [00:36, 959.35 examples/s]35912 examples [00:37, 950.90 examples/s]36012 examples [00:37, 964.38 examples/s]36111 examples [00:37, 969.08 examples/s]36208 examples [00:37, 945.47 examples/s]36304 examples [00:37, 946.72 examples/s]36411 examples [00:37, 978.26 examples/s]36510 examples [00:37, 972.29 examples/s]36609 examples [00:37, 976.95 examples/s]36715 examples [00:37, 998.20 examples/s]36818 examples [00:37, 1005.12 examples/s]36919 examples [00:38, 991.73 examples/s] 37022 examples [00:38, 1000.81 examples/s]37124 examples [00:38, 1003.68 examples/s]37225 examples [00:38, 996.84 examples/s] 37325 examples [00:38, 972.95 examples/s]37423 examples [00:38, 966.87 examples/s]37520 examples [00:38, 947.77 examples/s]37618 examples [00:38, 954.40 examples/s]37714 examples [00:38, 953.83 examples/s]37810 examples [00:39, 933.97 examples/s]37904 examples [00:39, 926.23 examples/s]37997 examples [00:39, 926.29 examples/s]38090 examples [00:39, 914.87 examples/s]38183 examples [00:39, 918.92 examples/s]38275 examples [00:39, 891.42 examples/s]38365 examples [00:39, 886.00 examples/s]38463 examples [00:39, 910.79 examples/s]38560 examples [00:39, 927.68 examples/s]38655 examples [00:39, 932.70 examples/s]38753 examples [00:40, 943.94 examples/s]38848 examples [00:40, 941.24 examples/s]38943 examples [00:40, 926.33 examples/s]39040 examples [00:40, 936.38 examples/s]39136 examples [00:40, 942.21 examples/s]39239 examples [00:40, 966.58 examples/s]39336 examples [00:40, 967.02 examples/s]39433 examples [00:40, 963.36 examples/s]39532 examples [00:40, 969.19 examples/s]39637 examples [00:40, 989.98 examples/s]39737 examples [00:41, 991.61 examples/s]39839 examples [00:41, 998.26 examples/s]39939 examples [00:41, 981.25 examples/s]40038 examples [00:41, 935.08 examples/s]40138 examples [00:41, 953.54 examples/s]40240 examples [00:41, 970.83 examples/s]40350 examples [00:41, 1002.76 examples/s]40451 examples [00:41, 978.18 examples/s] 40550 examples [00:41, 973.32 examples/s]40648 examples [00:41, 962.67 examples/s]40747 examples [00:42, 970.32 examples/s]40845 examples [00:42, 956.14 examples/s]40941 examples [00:42, 921.82 examples/s]41034 examples [00:42, 924.11 examples/s]41138 examples [00:42, 956.01 examples/s]41236 examples [00:42, 961.98 examples/s]41336 examples [00:42, 970.34 examples/s]41434 examples [00:42, 966.36 examples/s]41531 examples [00:42, 964.29 examples/s]41629 examples [00:43, 967.10 examples/s]41729 examples [00:43, 975.80 examples/s]41827 examples [00:43, 974.77 examples/s]41925 examples [00:43, 959.98 examples/s]42022 examples [00:43, 953.76 examples/s]42120 examples [00:43, 959.01 examples/s]42218 examples [00:43, 964.18 examples/s]42315 examples [00:43, 962.68 examples/s]42412 examples [00:43, 924.46 examples/s]42512 examples [00:43, 945.76 examples/s]42611 examples [00:44, 956.91 examples/s]42708 examples [00:44, 958.55 examples/s]42805 examples [00:44, 945.93 examples/s]42900 examples [00:44, 929.29 examples/s]43000 examples [00:44, 947.35 examples/s]43099 examples [00:44, 956.69 examples/s]43195 examples [00:44, 944.74 examples/s]43290 examples [00:44, 930.08 examples/s]43388 examples [00:44, 943.68 examples/s]43484 examples [00:44, 945.90 examples/s]43579 examples [00:45, 945.68 examples/s]43679 examples [00:45, 960.61 examples/s]43778 examples [00:45, 967.21 examples/s]43879 examples [00:45, 978.19 examples/s]43985 examples [00:45, 999.60 examples/s]44093 examples [00:45, 1019.55 examples/s]44200 examples [00:45, 1032.70 examples/s]44304 examples [00:45, 1029.17 examples/s]44408 examples [00:45, 1020.89 examples/s]44514 examples [00:45, 1031.61 examples/s]44618 examples [00:46, 1022.82 examples/s]44721 examples [00:46, 1005.91 examples/s]44822 examples [00:46, 989.19 examples/s] 44922 examples [00:46, 973.47 examples/s]45020 examples [00:46, 970.98 examples/s]45118 examples [00:46, 962.85 examples/s]45215 examples [00:46, 946.86 examples/s]45310 examples [00:46, 939.21 examples/s]45410 examples [00:46, 954.52 examples/s]45506 examples [00:47, 948.66 examples/s]45609 examples [00:47, 968.63 examples/s]45714 examples [00:47, 991.50 examples/s]45814 examples [00:47, 972.12 examples/s]45912 examples [00:47, 968.48 examples/s]46010 examples [00:47, 970.25 examples/s]46119 examples [00:47, 1002.95 examples/s]46224 examples [00:47, 1016.09 examples/s]46326 examples [00:47, 1014.56 examples/s]46428 examples [00:47, 1001.14 examples/s]46531 examples [00:48, 1006.87 examples/s]46632 examples [00:48, 992.61 examples/s] 46732 examples [00:48, 973.27 examples/s]46831 examples [00:48, 977.85 examples/s]46937 examples [00:48, 999.95 examples/s]47038 examples [00:48, 994.72 examples/s]47140 examples [00:48, 1001.24 examples/s]47241 examples [00:48, 993.67 examples/s] 47341 examples [00:48, 969.71 examples/s]47439 examples [00:48, 961.87 examples/s]47538 examples [00:49, 968.39 examples/s]47640 examples [00:49, 982.97 examples/s]47740 examples [00:49, 985.46 examples/s]47839 examples [00:49, 976.17 examples/s]47954 examples [00:49, 1022.13 examples/s]48058 examples [00:49, 1025.78 examples/s]48162 examples [00:49, 1022.82 examples/s]48265 examples [00:49, 1009.06 examples/s]48367 examples [00:49, 968.51 examples/s] 48465 examples [00:50, 937.18 examples/s]48560 examples [00:50, 932.95 examples/s]48663 examples [00:50, 958.02 examples/s]48767 examples [00:50, 979.47 examples/s]48866 examples [00:50, 971.57 examples/s]48965 examples [00:50, 975.82 examples/s]49063 examples [00:50, 969.74 examples/s]49161 examples [00:50, 965.53 examples/s]49261 examples [00:50, 973.87 examples/s]49359 examples [00:50, 971.09 examples/s]49458 examples [00:51, 973.27 examples/s]49559 examples [00:51, 983.94 examples/s]49658 examples [00:51, 972.86 examples/s]49761 examples [00:51, 987.85 examples/s]49860 examples [00:51, 959.47 examples/s]49964 examples [00:51, 981.64 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6865/50000 [00:00<00:00, 68636.97 examples/s] 39%|███▊      | 19289/50000 [00:00<00:00, 79281.15 examples/s] 62%|██████▏   | 31083/50000 [00:00<00:00, 87925.85 examples/s] 86%|████████▌ | 42769/50000 [00:00<00:00, 94976.22 examples/s]                                                               0 examples [00:00, ? examples/s]87 examples [00:00, 868.00 examples/s]186 examples [00:00, 900.52 examples/s]287 examples [00:00, 929.42 examples/s]376 examples [00:00, 916.09 examples/s]473 examples [00:00, 930.46 examples/s]566 examples [00:00, 929.04 examples/s]662 examples [00:00, 937.68 examples/s]753 examples [00:00, 927.64 examples/s]847 examples [00:00, 929.80 examples/s]940 examples [00:01, 928.04 examples/s]1032 examples [00:01, 925.27 examples/s]1124 examples [00:01, 923.29 examples/s]1216 examples [00:01, 901.64 examples/s]1308 examples [00:01, 904.76 examples/s]1404 examples [00:01, 918.32 examples/s]1503 examples [00:01, 938.42 examples/s]1602 examples [00:01, 953.27 examples/s]1701 examples [00:01, 960.95 examples/s]1798 examples [00:01, 957.97 examples/s]1905 examples [00:02, 987.75 examples/s]2008 examples [00:02, 1000.02 examples/s]2112 examples [00:02, 1011.63 examples/s]2214 examples [00:02, 1005.94 examples/s]2315 examples [00:02, 981.36 examples/s] 2414 examples [00:02, 927.93 examples/s]2509 examples [00:02, 934.41 examples/s]2615 examples [00:02, 968.55 examples/s]2716 examples [00:02, 977.48 examples/s]2815 examples [00:02, 930.03 examples/s]2909 examples [00:03, 921.27 examples/s]3002 examples [00:03, 922.17 examples/s]3095 examples [00:03, 921.43 examples/s]3188 examples [00:03, 905.49 examples/s]3288 examples [00:03, 930.80 examples/s]3386 examples [00:03, 941.54 examples/s]3486 examples [00:03, 956.11 examples/s]3583 examples [00:03, 958.92 examples/s]3680 examples [00:03, 950.14 examples/s]3782 examples [00:03, 968.16 examples/s]3880 examples [00:04, 959.68 examples/s]3977 examples [00:04, 943.97 examples/s]4072 examples [00:04, 923.29 examples/s]4165 examples [00:04, 923.75 examples/s]4262 examples [00:04, 935.48 examples/s]4356 examples [00:04, 930.67 examples/s]4452 examples [00:04, 936.45 examples/s]4547 examples [00:04, 939.44 examples/s]4642 examples [00:04, 933.87 examples/s]4736 examples [00:05, 931.19 examples/s]4830 examples [00:05, 922.16 examples/s]4923 examples [00:05, 923.70 examples/s]5019 examples [00:05, 932.23 examples/s]5113 examples [00:05, 910.72 examples/s]5205 examples [00:05, 884.62 examples/s]5297 examples [00:05, 893.15 examples/s]5387 examples [00:05, 889.74 examples/s]5478 examples [00:05, 893.87 examples/s]5568 examples [00:05, 869.45 examples/s]5656 examples [00:06, 864.26 examples/s]5743 examples [00:06, 818.37 examples/s]5830 examples [00:06, 831.40 examples/s]5922 examples [00:06, 855.68 examples/s]6012 examples [00:06, 868.42 examples/s]6107 examples [00:06, 890.78 examples/s]6207 examples [00:06, 917.43 examples/s]6302 examples [00:06, 925.67 examples/s]6398 examples [00:06, 932.72 examples/s]6492 examples [00:06, 930.84 examples/s]6590 examples [00:07, 944.47 examples/s]6687 examples [00:07, 951.97 examples/s]6791 examples [00:07, 974.70 examples/s]6889 examples [00:07, 945.76 examples/s]6995 examples [00:07, 977.35 examples/s]7095 examples [00:07, 983.24 examples/s]7194 examples [00:07, 967.94 examples/s]7293 examples [00:07, 972.95 examples/s]7402 examples [00:07, 1005.05 examples/s]7506 examples [00:08, 1013.82 examples/s]7608 examples [00:08, 952.56 examples/s] 7705 examples [00:08, 948.87 examples/s]7801 examples [00:08, 939.74 examples/s]7904 examples [00:08, 962.55 examples/s]8008 examples [00:08, 983.99 examples/s]8111 examples [00:08, 995.94 examples/s]8222 examples [00:08, 1026.91 examples/s]8330 examples [00:08, 1041.53 examples/s]8435 examples [00:08, 1016.01 examples/s]8541 examples [00:09, 1027.99 examples/s]8645 examples [00:09, 1022.25 examples/s]8748 examples [00:09, 1021.39 examples/s]8852 examples [00:09, 1025.03 examples/s]8955 examples [00:09, 1010.00 examples/s]9061 examples [00:09, 1022.57 examples/s]9165 examples [00:09, 1026.47 examples/s]9268 examples [00:09, 1021.78 examples/s]9375 examples [00:09, 1033.25 examples/s]9479 examples [00:09, 1019.96 examples/s]9583 examples [00:10, 1025.79 examples/s]9686 examples [00:10, 1014.09 examples/s]9788 examples [00:10, 1002.69 examples/s]9889 examples [00:10, 1000.81 examples/s]9990 examples [00:10, 954.11 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1VWGI2/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete1VWGI2/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8efe6cc0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8efe6cc0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8efe6cc0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f4a0ab828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8f69bbf048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f8efe6cc0d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f8efe6cc0d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f8efe6cc0d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8e8493f5c0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8efc868550>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.29 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.18 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 13.18 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.71 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 11.71 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.71 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.71 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.20 file/s]2020-08-18 06:08:33.664011: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-18 06:08:33.667222: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-18 06:08:33.667360: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x564fb0cc3e50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-18 06:08:33.667375: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 474665.07it/s] 85%|████████▌ | 8470528/9912422 [00:00<00:02, 676445.62it/s]9920512it [00:00, 45866514.36it/s]                           
0it [00:00, ?it/s]32768it [00:00, 590259.41it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1045393.08it/s]1654784it [00:00, 12138426.23it/s]                           
0it [00:00, ?it/s]8192it [00:00, 255240.71it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

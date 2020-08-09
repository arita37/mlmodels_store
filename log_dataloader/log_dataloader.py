
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f6b1c12d510> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f6b1c12d510> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f6b86ddd488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f6b86ddd488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f6ba5ffbea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f6ba5ffbea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6b34116158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6b34116158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6b34116158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▉      | 3907584/9912422 [00:00<00:00, 35087979.45it/s]9920512it [00:00, 36777843.62it/s]                             
0it [00:00, ?it/s]32768it [00:00, 568265.35it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 137450.77it/s]1654784it [00:00, 9975648.46it/s]                          
0it [00:00, ?it/s]8192it [00:00, 156684.15it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6b1b3686d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6b32519940>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f6b3410dd90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f6b3410dd90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f6b3410dd90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:56,  2.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:56,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:55,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:55,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:55,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:54,  2.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:39,  4.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:39,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:38,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:38,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:38,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:38,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:37,  4.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   7%|▋         | 12/162 [00:00<00:27,  5.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:27,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:26,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:25,  5.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:00<00:18,  7.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:18,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:18,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:18,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:18,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:18,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:18,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:17,  7.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:13, 10.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:13, 10.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:12, 10.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:12, 10.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:12, 10.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:12, 10.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:12, 10.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:12, 10.42 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:09, 13.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:09, 13.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:09, 13.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:09, 13.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:09, 13.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:08, 13.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:08, 13.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:08, 13.92 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▍       | 40/162 [00:00<00:06, 18.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:06, 18.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:04, 23.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:04, 23.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:04, 23.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:04, 23.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:04, 23.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:04, 23.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:04, 23.28 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 28.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:03, 28.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:03, 28.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:03, 28.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:03, 28.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:03, 28.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:03, 28.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:03, 28.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 34.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:02, 34.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:02, 34.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:02, 34.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:02, 34.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:02, 34.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:02, 34.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:02, 34.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 39.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:02, 39.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 39.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 39.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 39.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 39.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 39.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 39.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 44.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:01, 44.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:01, 44.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:01, 44.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:01, 44.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:01, 44.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:01, 44.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 44.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 48.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 48.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 48.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 48.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 48.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 48.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 48.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 51.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 51.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 51.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 51.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 51.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 51.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 51.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 54.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 54.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 54.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 54.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 56.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 56.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 56.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 56.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 56.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 56.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 56.72 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 58.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 58.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 58.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 59.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 59.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 59.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 59.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 59.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 59.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 59.26 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 60.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 60.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.64 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.41 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 61.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 61.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 61.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 61.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 61.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 61.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 61.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 61.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 61.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 61.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 61.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 61.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 61.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 61.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 61.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 61.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 63.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 63.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 62.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 62.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 62.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 62.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.92s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.92s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.92s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 62.75 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.39s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.92s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 62.75 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.39s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.39s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.05 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.39s/ url]
0 examples [00:00, ? examples/s]2020-08-09 00:08:17.161158: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-09 00:08:17.174996: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-09 00:08:17.175202: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x562a4c22d260 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-09 00:08:17.175224: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
24 examples [00:00, 237.50 examples/s]112 examples [00:00, 303.80 examples/s]200 examples [00:00, 377.38 examples/s]283 examples [00:00, 450.68 examples/s]369 examples [00:00, 524.86 examples/s]452 examples [00:00, 589.65 examples/s]535 examples [00:00, 645.43 examples/s]622 examples [00:00, 699.26 examples/s]714 examples [00:00, 753.06 examples/s]804 examples [00:01, 789.98 examples/s]889 examples [00:01, 793.39 examples/s]978 examples [00:01, 818.32 examples/s]1066 examples [00:01, 832.98 examples/s]1152 examples [00:01, 835.86 examples/s]1238 examples [00:01, 838.57 examples/s]1323 examples [00:01, 818.35 examples/s]1406 examples [00:01, 815.43 examples/s]1489 examples [00:01, 815.89 examples/s]1571 examples [00:01, 814.91 examples/s]1653 examples [00:02, 786.41 examples/s]1741 examples [00:02, 811.07 examples/s]1831 examples [00:02, 833.44 examples/s]1917 examples [00:02, 840.04 examples/s]2005 examples [00:02, 849.88 examples/s]2094 examples [00:02, 860.12 examples/s]2181 examples [00:02, 857.91 examples/s]2267 examples [00:02, 844.21 examples/s]2352 examples [00:02, 845.60 examples/s]2437 examples [00:02, 837.53 examples/s]2527 examples [00:03, 852.97 examples/s]2618 examples [00:03, 868.11 examples/s]2710 examples [00:03, 882.71 examples/s]2799 examples [00:03, 848.60 examples/s]2885 examples [00:03, 841.51 examples/s]2971 examples [00:03, 844.25 examples/s]3057 examples [00:03, 847.80 examples/s]3145 examples [00:03, 854.34 examples/s]3233 examples [00:03, 860.12 examples/s]3320 examples [00:03, 860.25 examples/s]3409 examples [00:04, 867.40 examples/s]3496 examples [00:04, 855.63 examples/s]3582 examples [00:04, 789.46 examples/s]3669 examples [00:04, 811.30 examples/s]3764 examples [00:04, 846.23 examples/s]3852 examples [00:04, 855.49 examples/s]3939 examples [00:04, 853.61 examples/s]4028 examples [00:04, 863.74 examples/s]4115 examples [00:04, 863.00 examples/s]4204 examples [00:05, 868.57 examples/s]4292 examples [00:05, 865.34 examples/s]4379 examples [00:05, 863.52 examples/s]4469 examples [00:05, 873.84 examples/s]4557 examples [00:05, 869.44 examples/s]4645 examples [00:05, 857.44 examples/s]4731 examples [00:05, 845.61 examples/s]4816 examples [00:05, 846.17 examples/s]4907 examples [00:05, 864.17 examples/s]4994 examples [00:05, 855.03 examples/s]5082 examples [00:06, 859.75 examples/s]5169 examples [00:06, 844.36 examples/s]5255 examples [00:06, 846.85 examples/s]5344 examples [00:06, 858.72 examples/s]5434 examples [00:06, 868.72 examples/s]5526 examples [00:06, 881.62 examples/s]5617 examples [00:06, 888.72 examples/s]5706 examples [00:06, 885.07 examples/s]5795 examples [00:06, 886.21 examples/s]5884 examples [00:06, 884.60 examples/s]5973 examples [00:07, 876.88 examples/s]6061 examples [00:07, 870.10 examples/s]6152 examples [00:07, 881.30 examples/s]6244 examples [00:07, 890.54 examples/s]6334 examples [00:07, 878.57 examples/s]6427 examples [00:07, 891.50 examples/s]6518 examples [00:07, 896.55 examples/s]6608 examples [00:07, 892.67 examples/s]6698 examples [00:07, 885.45 examples/s]6787 examples [00:07, 878.54 examples/s]6880 examples [00:08, 891.42 examples/s]6971 examples [00:08, 896.11 examples/s]7062 examples [00:08, 898.45 examples/s]7153 examples [00:08, 900.91 examples/s]7244 examples [00:08, 888.85 examples/s]7336 examples [00:08, 896.58 examples/s]7426 examples [00:08, 888.26 examples/s]7515 examples [00:08, 862.69 examples/s]7603 examples [00:08, 866.51 examples/s]7690 examples [00:09, 865.36 examples/s]7777 examples [00:09, 818.12 examples/s]7864 examples [00:09, 831.08 examples/s]7950 examples [00:09, 838.22 examples/s]8035 examples [00:09, 834.09 examples/s]8124 examples [00:09, 847.90 examples/s]8213 examples [00:09, 857.71 examples/s]8300 examples [00:09, 860.63 examples/s]8387 examples [00:09, 853.46 examples/s]8473 examples [00:09, 850.81 examples/s]8559 examples [00:10, 849.18 examples/s]8644 examples [00:10, 821.87 examples/s]8727 examples [00:10, 807.66 examples/s]8808 examples [00:10, 793.39 examples/s]8894 examples [00:10, 811.10 examples/s]8976 examples [00:10, 812.43 examples/s]9065 examples [00:10, 832.36 examples/s]9152 examples [00:10, 842.80 examples/s]9239 examples [00:10, 848.92 examples/s]9325 examples [00:10, 837.46 examples/s]9416 examples [00:11, 856.01 examples/s]9510 examples [00:11, 878.66 examples/s]9601 examples [00:11, 886.22 examples/s]9692 examples [00:11, 892.54 examples/s]9784 examples [00:11, 898.64 examples/s]9875 examples [00:11, 901.52 examples/s]9966 examples [00:11, 885.45 examples/s]10055 examples [00:11, 834.08 examples/s]10148 examples [00:11, 858.47 examples/s]10235 examples [00:12, 827.41 examples/s]10321 examples [00:12, 836.71 examples/s]10407 examples [00:12, 842.73 examples/s]10496 examples [00:12, 856.35 examples/s]10588 examples [00:12, 871.92 examples/s]10679 examples [00:12, 882.53 examples/s]10768 examples [00:12, 883.32 examples/s]10858 examples [00:12, 886.24 examples/s]10949 examples [00:12, 891.93 examples/s]11039 examples [00:12, 893.71 examples/s]11129 examples [00:13, 867.01 examples/s]11219 examples [00:13, 875.91 examples/s]11307 examples [00:13, 855.86 examples/s]11393 examples [00:13, 840.59 examples/s]11478 examples [00:13, 824.22 examples/s]11564 examples [00:13, 832.16 examples/s]11648 examples [00:13, 829.46 examples/s]11732 examples [00:13, 809.79 examples/s]11823 examples [00:13, 836.68 examples/s]11908 examples [00:13, 811.38 examples/s]11995 examples [00:14, 827.80 examples/s]12079 examples [00:14, 830.15 examples/s]12165 examples [00:14, 838.28 examples/s]12250 examples [00:14, 835.56 examples/s]12335 examples [00:14, 837.56 examples/s]12425 examples [00:14, 854.34 examples/s]12514 examples [00:14, 861.43 examples/s]12601 examples [00:14, 819.07 examples/s]12688 examples [00:14, 833.13 examples/s]12773 examples [00:15, 835.91 examples/s]12860 examples [00:15, 845.11 examples/s]12950 examples [00:15, 860.64 examples/s]13043 examples [00:15, 879.52 examples/s]13135 examples [00:15, 888.53 examples/s]13225 examples [00:15, 891.57 examples/s]13315 examples [00:15, 882.65 examples/s]13404 examples [00:15, 884.57 examples/s]13494 examples [00:15, 887.58 examples/s]13584 examples [00:15, 889.25 examples/s]13673 examples [00:16, 885.19 examples/s]13762 examples [00:16, 882.17 examples/s]13851 examples [00:16, 872.00 examples/s]13939 examples [00:16, 850.21 examples/s]14025 examples [00:16, 842.74 examples/s]14110 examples [00:16, 844.62 examples/s]14196 examples [00:16, 848.37 examples/s]14281 examples [00:16, 829.62 examples/s]14365 examples [00:16, 823.16 examples/s]14448 examples [00:16, 799.52 examples/s]14529 examples [00:17, 787.99 examples/s]14608 examples [00:17, 782.73 examples/s]14691 examples [00:17, 793.29 examples/s]14771 examples [00:17, 790.85 examples/s]14852 examples [00:17, 794.86 examples/s]14934 examples [00:17, 800.09 examples/s]15015 examples [00:17, 796.99 examples/s]15097 examples [00:17, 802.99 examples/s]15178 examples [00:17, 804.87 examples/s]15263 examples [00:17, 816.35 examples/s]15345 examples [00:18, 806.81 examples/s]15432 examples [00:18, 822.73 examples/s]15518 examples [00:18, 832.77 examples/s]15602 examples [00:18, 832.76 examples/s]15689 examples [00:18, 840.14 examples/s]15774 examples [00:18, 820.17 examples/s]15857 examples [00:18, 817.15 examples/s]15939 examples [00:18, 801.89 examples/s]16020 examples [00:18, 799.94 examples/s]16101 examples [00:19, 786.20 examples/s]16180 examples [00:19, 769.80 examples/s]16258 examples [00:19, 761.74 examples/s]16335 examples [00:19, 762.34 examples/s]16417 examples [00:19, 776.83 examples/s]16495 examples [00:19, 753.94 examples/s]16574 examples [00:19, 763.77 examples/s]16661 examples [00:19, 790.53 examples/s]16750 examples [00:19, 815.45 examples/s]16833 examples [00:19, 808.37 examples/s]16915 examples [00:20, 804.83 examples/s]16996 examples [00:20, 787.32 examples/s]17075 examples [00:20, 769.42 examples/s]17157 examples [00:20, 783.82 examples/s]17241 examples [00:20, 798.11 examples/s]17322 examples [00:20, 798.53 examples/s]17403 examples [00:20, 799.78 examples/s]17484 examples [00:20, 798.85 examples/s]17564 examples [00:20, 796.59 examples/s]17644 examples [00:20, 773.91 examples/s]17725 examples [00:21, 784.34 examples/s]17806 examples [00:21, 790.77 examples/s]17890 examples [00:21, 802.49 examples/s]17974 examples [00:21, 811.34 examples/s]18057 examples [00:21, 812.81 examples/s]18139 examples [00:21, 811.86 examples/s]18223 examples [00:21, 817.36 examples/s]18305 examples [00:21, 809.59 examples/s]18395 examples [00:21, 832.98 examples/s]18484 examples [00:21, 847.38 examples/s]18570 examples [00:22, 850.72 examples/s]18656 examples [00:22, 838.08 examples/s]18740 examples [00:22, 821.35 examples/s]18823 examples [00:22, 820.93 examples/s]18906 examples [00:22, 819.81 examples/s]18996 examples [00:22, 841.08 examples/s]19081 examples [00:22, 836.74 examples/s]19165 examples [00:22, 812.16 examples/s]19251 examples [00:22, 824.17 examples/s]19338 examples [00:23, 835.48 examples/s]19422 examples [00:23, 833.07 examples/s]19508 examples [00:23, 840.95 examples/s]19593 examples [00:23, 842.36 examples/s]19678 examples [00:23, 835.76 examples/s]19767 examples [00:23, 849.26 examples/s]19853 examples [00:23, 846.71 examples/s]19938 examples [00:23, 836.49 examples/s]20022 examples [00:23, 768.42 examples/s]20109 examples [00:23, 795.18 examples/s]20190 examples [00:24, 795.61 examples/s]20271 examples [00:24, 799.46 examples/s]20355 examples [00:24, 809.05 examples/s]20442 examples [00:24, 825.96 examples/s]20530 examples [00:24, 840.84 examples/s]20617 examples [00:24, 849.30 examples/s]20704 examples [00:24, 853.01 examples/s]20793 examples [00:24, 863.17 examples/s]20880 examples [00:24, 859.86 examples/s]20967 examples [00:24, 855.42 examples/s]21056 examples [00:25, 863.27 examples/s]21143 examples [00:25, 848.44 examples/s]21228 examples [00:25, 838.81 examples/s]21314 examples [00:25, 843.98 examples/s]21401 examples [00:25, 850.91 examples/s]21488 examples [00:25, 853.48 examples/s]21576 examples [00:25, 861.11 examples/s]21667 examples [00:25, 871.25 examples/s]21758 examples [00:25, 881.21 examples/s]21848 examples [00:25, 884.52 examples/s]21937 examples [00:26, 880.87 examples/s]22026 examples [00:26, 879.52 examples/s]22116 examples [00:26, 884.15 examples/s]22207 examples [00:26, 891.37 examples/s]22300 examples [00:26, 900.40 examples/s]22391 examples [00:26, 885.64 examples/s]22480 examples [00:26, 881.01 examples/s]22569 examples [00:26, 880.98 examples/s]22658 examples [00:26, 877.31 examples/s]22746 examples [00:27, 831.67 examples/s]22835 examples [00:27, 847.86 examples/s]22922 examples [00:27, 853.68 examples/s]23008 examples [00:27, 854.83 examples/s]23094 examples [00:27, 849.45 examples/s]23185 examples [00:27, 864.83 examples/s]23277 examples [00:27, 878.10 examples/s]23365 examples [00:27, 874.92 examples/s]23454 examples [00:27, 877.04 examples/s]23542 examples [00:27, 871.91 examples/s]23630 examples [00:28, 863.49 examples/s]23721 examples [00:28, 874.74 examples/s]23812 examples [00:28, 882.56 examples/s]23901 examples [00:28, 882.14 examples/s]23990 examples [00:28, 880.49 examples/s]24079 examples [00:28, 869.83 examples/s]24167 examples [00:28, 863.87 examples/s]24254 examples [00:28, 863.61 examples/s]24341 examples [00:28, 863.59 examples/s]24428 examples [00:28, 860.55 examples/s]24517 examples [00:29, 866.70 examples/s]24604 examples [00:29, 856.22 examples/s]24692 examples [00:29, 862.34 examples/s]24781 examples [00:29, 866.28 examples/s]24871 examples [00:29, 876.06 examples/s]24962 examples [00:29, 884.97 examples/s]25051 examples [00:29, 881.52 examples/s]25140 examples [00:29, 876.65 examples/s]25228 examples [00:29, 853.60 examples/s]25314 examples [00:29, 830.19 examples/s]25399 examples [00:30, 835.45 examples/s]25489 examples [00:30, 851.22 examples/s]25577 examples [00:30, 857.74 examples/s]25663 examples [00:30, 850.63 examples/s]25752 examples [00:30, 860.69 examples/s]25839 examples [00:30, 845.23 examples/s]25924 examples [00:30, 835.33 examples/s]26012 examples [00:30, 847.85 examples/s]26100 examples [00:30, 855.36 examples/s]26190 examples [00:31, 866.30 examples/s]26282 examples [00:31, 880.14 examples/s]26371 examples [00:31, 856.90 examples/s]26461 examples [00:31, 868.73 examples/s]26549 examples [00:31, 840.54 examples/s]26634 examples [00:31, 806.04 examples/s]26726 examples [00:31, 836.73 examples/s]26814 examples [00:31, 846.86 examples/s]26904 examples [00:31, 860.12 examples/s]26993 examples [00:31, 867.18 examples/s]27083 examples [00:32, 874.33 examples/s]27173 examples [00:32, 879.09 examples/s]27262 examples [00:32, 880.89 examples/s]27353 examples [00:32, 888.52 examples/s]27442 examples [00:32, 881.37 examples/s]27534 examples [00:32, 891.51 examples/s]27624 examples [00:32, 883.84 examples/s]27713 examples [00:32, 835.52 examples/s]27798 examples [00:32, 832.32 examples/s]27886 examples [00:32, 844.34 examples/s]27977 examples [00:33, 862.43 examples/s]28066 examples [00:33, 869.59 examples/s]28156 examples [00:33, 877.42 examples/s]28244 examples [00:33, 861.63 examples/s]28332 examples [00:33, 865.49 examples/s]28424 examples [00:33, 881.09 examples/s]28515 examples [00:33, 887.99 examples/s]28604 examples [00:33, 868.38 examples/s]28694 examples [00:33, 875.38 examples/s]28782 examples [00:34, 869.03 examples/s]28873 examples [00:34, 880.07 examples/s]28962 examples [00:34, 882.58 examples/s]29051 examples [00:34, 882.95 examples/s]29140 examples [00:34, 868.85 examples/s]29227 examples [00:34, 854.24 examples/s]29315 examples [00:34, 861.05 examples/s]29406 examples [00:34, 872.88 examples/s]29494 examples [00:34, 869.15 examples/s]29585 examples [00:34, 879.57 examples/s]29674 examples [00:35, 879.81 examples/s]29763 examples [00:35, 864.75 examples/s]29850 examples [00:35, 851.28 examples/s]29938 examples [00:35, 856.88 examples/s]30024 examples [00:35, 784.62 examples/s]30111 examples [00:35, 806.98 examples/s]30198 examples [00:35, 824.68 examples/s]30287 examples [00:35, 840.79 examples/s]30377 examples [00:35, 855.62 examples/s]30466 examples [00:35, 863.20 examples/s]30553 examples [00:36, 864.38 examples/s]30644 examples [00:36, 877.41 examples/s]30742 examples [00:36, 905.69 examples/s]30833 examples [00:36, 896.26 examples/s]30923 examples [00:36, 897.37 examples/s]31013 examples [00:36, 887.33 examples/s]31102 examples [00:36, 883.25 examples/s]31191 examples [00:36, 884.59 examples/s]31280 examples [00:36, 876.16 examples/s]31368 examples [00:36, 875.01 examples/s]31459 examples [00:37, 884.32 examples/s]31548 examples [00:37, 875.69 examples/s]31642 examples [00:37, 891.74 examples/s]31732 examples [00:37, 888.79 examples/s]31821 examples [00:37, 880.50 examples/s]31910 examples [00:37, 864.57 examples/s]32002 examples [00:37, 878.62 examples/s]32094 examples [00:37, 888.87 examples/s]32184 examples [00:37, 890.29 examples/s]32277 examples [00:38, 889.30 examples/s]32366 examples [00:38, 876.68 examples/s]32458 examples [00:38, 887.47 examples/s]32550 examples [00:38, 894.71 examples/s]32640 examples [00:38, 891.75 examples/s]32731 examples [00:38, 895.68 examples/s]32821 examples [00:38, 892.34 examples/s]32911 examples [00:38, 893.51 examples/s]33001 examples [00:38, 876.09 examples/s]33089 examples [00:38, 868.49 examples/s]33178 examples [00:39, 873.90 examples/s]33266 examples [00:39, 829.14 examples/s]33358 examples [00:39, 853.72 examples/s]33448 examples [00:39, 866.35 examples/s]33536 examples [00:39, 868.64 examples/s]33624 examples [00:39, 867.95 examples/s]33712 examples [00:39, 842.83 examples/s]33800 examples [00:39, 851.76 examples/s]33887 examples [00:39, 855.09 examples/s]33974 examples [00:39, 857.49 examples/s]34063 examples [00:40, 865.50 examples/s]34150 examples [00:40, 856.19 examples/s]34238 examples [00:40, 860.87 examples/s]34325 examples [00:40, 860.14 examples/s]34412 examples [00:40, 859.50 examples/s]34498 examples [00:40, 854.10 examples/s]34590 examples [00:40, 871.07 examples/s]34680 examples [00:40, 877.38 examples/s]34768 examples [00:40, 868.23 examples/s]34859 examples [00:40, 879.85 examples/s]34949 examples [00:41, 884.57 examples/s]35039 examples [00:41, 888.35 examples/s]35130 examples [00:41, 894.15 examples/s]35220 examples [00:41, 887.52 examples/s]35309 examples [00:41, 880.76 examples/s]35398 examples [00:41, 868.11 examples/s]35485 examples [00:41, 867.83 examples/s]35573 examples [00:41, 871.37 examples/s]35663 examples [00:41, 876.63 examples/s]35752 examples [00:42, 880.06 examples/s]35841 examples [00:42, 879.24 examples/s]35932 examples [00:42, 887.72 examples/s]36021 examples [00:42, 887.75 examples/s]36110 examples [00:42, 880.79 examples/s]36200 examples [00:42, 883.66 examples/s]36289 examples [00:42, 881.60 examples/s]36378 examples [00:42, 877.27 examples/s]36466 examples [00:42, 874.27 examples/s]36554 examples [00:42, 857.10 examples/s]36640 examples [00:43, 853.60 examples/s]36726 examples [00:43, 844.11 examples/s]36811 examples [00:43, 829.99 examples/s]36899 examples [00:43, 844.37 examples/s]36984 examples [00:43, 844.61 examples/s]37071 examples [00:43, 851.17 examples/s]37159 examples [00:43, 857.90 examples/s]37245 examples [00:43, 857.34 examples/s]37331 examples [00:43, 840.78 examples/s]37416 examples [00:43, 832.51 examples/s]37505 examples [00:44, 847.93 examples/s]37596 examples [00:44, 865.05 examples/s]37683 examples [00:44, 857.04 examples/s]37773 examples [00:44, 869.49 examples/s]37861 examples [00:44, 872.49 examples/s]37950 examples [00:44, 876.06 examples/s]38038 examples [00:44, 874.45 examples/s]38128 examples [00:44, 881.09 examples/s]38218 examples [00:44, 886.33 examples/s]38307 examples [00:44, 879.95 examples/s]38398 examples [00:45, 887.38 examples/s]38487 examples [00:45, 884.13 examples/s]38576 examples [00:45, 883.08 examples/s]38666 examples [00:45, 886.11 examples/s]38755 examples [00:45, 882.69 examples/s]38844 examples [00:45, 792.96 examples/s]38926 examples [00:45, 788.86 examples/s]39007 examples [00:45, 762.02 examples/s]39085 examples [00:45, 758.29 examples/s]39163 examples [00:46, 762.59 examples/s]39241 examples [00:46, 767.29 examples/s]39319 examples [00:46, 766.95 examples/s]39400 examples [00:46, 778.89 examples/s]39484 examples [00:46, 793.25 examples/s]39564 examples [00:46, 770.61 examples/s]39649 examples [00:46, 790.36 examples/s]39739 examples [00:46, 819.83 examples/s]39827 examples [00:46, 836.99 examples/s]39914 examples [00:46, 846.49 examples/s]39999 examples [00:47, 847.22 examples/s]40084 examples [00:47, 786.09 examples/s]40173 examples [00:47, 812.11 examples/s]40261 examples [00:47, 828.95 examples/s]40347 examples [00:47, 837.86 examples/s]40432 examples [00:47, 841.01 examples/s]40520 examples [00:47, 851.66 examples/s]40606 examples [00:47, 851.55 examples/s]40693 examples [00:47, 856.53 examples/s]40779 examples [00:47, 842.23 examples/s]40866 examples [00:48, 848.22 examples/s]40952 examples [00:48, 850.65 examples/s]41040 examples [00:48, 858.68 examples/s]41129 examples [00:48, 867.37 examples/s]41216 examples [00:48, 848.15 examples/s]41304 examples [00:48, 857.03 examples/s]41393 examples [00:48, 865.20 examples/s]41483 examples [00:48, 875.32 examples/s]41572 examples [00:48, 877.54 examples/s]41662 examples [00:48, 883.90 examples/s]41752 examples [00:49, 888.02 examples/s]41841 examples [00:49, 881.01 examples/s]41930 examples [00:49, 876.05 examples/s]42018 examples [00:49, 869.03 examples/s]42105 examples [00:49, 867.07 examples/s]42193 examples [00:49, 869.13 examples/s]42282 examples [00:49, 872.12 examples/s]42370 examples [00:49, 859.90 examples/s]42457 examples [00:49, 840.76 examples/s]42544 examples [00:50, 846.95 examples/s]42639 examples [00:50, 873.81 examples/s]42731 examples [00:50, 885.47 examples/s]42823 examples [00:50, 893.42 examples/s]42913 examples [00:50, 887.18 examples/s]43004 examples [00:50, 892.37 examples/s]43095 examples [00:50, 896.25 examples/s]43187 examples [00:50, 901.92 examples/s]43278 examples [00:50, 874.15 examples/s]43367 examples [00:50, 878.03 examples/s]43457 examples [00:51, 882.77 examples/s]43546 examples [00:51, 884.22 examples/s]43636 examples [00:51, 888.27 examples/s]43725 examples [00:51, 875.90 examples/s]43813 examples [00:51, 874.19 examples/s]43904 examples [00:51, 882.51 examples/s]43994 examples [00:51, 886.25 examples/s]44083 examples [00:51, 885.39 examples/s]44172 examples [00:51, 880.59 examples/s]44263 examples [00:51, 887.89 examples/s]44353 examples [00:52, 890.23 examples/s]44443 examples [00:52, 891.11 examples/s]44533 examples [00:52, 854.70 examples/s]44625 examples [00:52, 872.37 examples/s]44713 examples [00:52, 874.43 examples/s]44801 examples [00:52, 864.83 examples/s]44888 examples [00:52, 853.85 examples/s]44974 examples [00:52, 845.90 examples/s]45065 examples [00:52, 863.38 examples/s]45152 examples [00:52, 864.50 examples/s]45242 examples [00:53, 874.56 examples/s]45332 examples [00:53, 880.90 examples/s]45421 examples [00:53, 861.25 examples/s]45508 examples [00:53, 862.18 examples/s]45599 examples [00:53, 875.58 examples/s]45687 examples [00:53, 856.82 examples/s]45773 examples [00:53, 839.99 examples/s]45860 examples [00:53, 848.53 examples/s]45948 examples [00:53, 856.27 examples/s]46037 examples [00:54, 865.41 examples/s]46126 examples [00:54, 871.24 examples/s]46218 examples [00:54, 885.17 examples/s]46309 examples [00:54, 890.12 examples/s]46399 examples [00:54, 893.06 examples/s]46489 examples [00:54, 889.47 examples/s]46579 examples [00:54, 890.44 examples/s]46672 examples [00:54, 901.45 examples/s]46763 examples [00:54, 893.42 examples/s]46853 examples [00:54, 884.53 examples/s]46942 examples [00:55, 871.22 examples/s]47030 examples [00:55, 860.63 examples/s]47117 examples [00:55, 861.89 examples/s]47206 examples [00:55, 867.59 examples/s]47295 examples [00:55, 874.11 examples/s]47385 examples [00:55, 880.46 examples/s]47474 examples [00:55, 881.42 examples/s]47567 examples [00:55, 894.85 examples/s]47657 examples [00:55, 891.35 examples/s]47748 examples [00:55, 896.33 examples/s]47839 examples [00:56, 899.29 examples/s]47930 examples [00:56, 898.76 examples/s]48023 examples [00:56, 906.46 examples/s]48114 examples [00:56, 893.91 examples/s]48205 examples [00:56, 897.45 examples/s]48295 examples [00:56, 874.78 examples/s]48385 examples [00:56, 879.05 examples/s]48474 examples [00:56, 869.65 examples/s]48562 examples [00:56, 857.48 examples/s]48653 examples [00:56, 871.50 examples/s]48742 examples [00:57, 874.68 examples/s]48830 examples [00:57, 875.21 examples/s]48921 examples [00:57, 884.38 examples/s]49010 examples [00:57, 884.79 examples/s]49099 examples [00:57, 855.99 examples/s]49189 examples [00:57, 867.87 examples/s]49276 examples [00:57, 863.36 examples/s]49365 examples [00:57, 868.04 examples/s]49454 examples [00:57, 874.26 examples/s]49542 examples [00:57, 874.90 examples/s]49630 examples [00:58, 871.91 examples/s]49719 examples [00:58, 875.20 examples/s]49812 examples [00:58, 890.81 examples/s]49902 examples [00:58, 886.11 examples/s]49992 examples [00:58, 890.00 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6575/50000 [00:00<00:00, 65745.83 examples/s] 35%|███▍      | 17297/50000 [00:00<00:00, 74376.40 examples/s] 57%|█████▋    | 28611/50000 [00:00<00:00, 82896.31 examples/s] 79%|███████▉  | 39594/50000 [00:00<00:00, 89479.20 examples/s]                                                               0 examples [00:00, ? examples/s]75 examples [00:00, 747.84 examples/s]166 examples [00:00, 788.43 examples/s]260 examples [00:00, 825.76 examples/s]352 examples [00:00, 850.05 examples/s]442 examples [00:00, 863.54 examples/s]533 examples [00:00, 874.86 examples/s]621 examples [00:00, 875.39 examples/s]713 examples [00:00, 886.23 examples/s]798 examples [00:00, 869.93 examples/s]888 examples [00:01, 877.56 examples/s]983 examples [00:01, 896.02 examples/s]1074 examples [00:01, 899.76 examples/s]1166 examples [00:01, 905.51 examples/s]1259 examples [00:01, 912.53 examples/s]1351 examples [00:01, 913.79 examples/s]1443 examples [00:01, 915.07 examples/s]1535 examples [00:01, 907.63 examples/s]1628 examples [00:01, 913.84 examples/s]1720 examples [00:01, 910.07 examples/s]1811 examples [00:02, 905.16 examples/s]1902 examples [00:02, 905.27 examples/s]1993 examples [00:02, 906.08 examples/s]2084 examples [00:02, 905.61 examples/s]2175 examples [00:02, 904.64 examples/s]2266 examples [00:02, 902.07 examples/s]2361 examples [00:02, 913.68 examples/s]2453 examples [00:02, 903.43 examples/s]2546 examples [00:02, 908.76 examples/s]2637 examples [00:02, 906.40 examples/s]2728 examples [00:03, 890.89 examples/s]2822 examples [00:03, 902.38 examples/s]2914 examples [00:03, 904.46 examples/s]3005 examples [00:03, 905.28 examples/s]3097 examples [00:03, 908.99 examples/s]3190 examples [00:03, 911.83 examples/s]3283 examples [00:03, 915.40 examples/s]3375 examples [00:03, 913.68 examples/s]3467 examples [00:03, 914.09 examples/s]3559 examples [00:03, 909.75 examples/s]3650 examples [00:04, 908.81 examples/s]3744 examples [00:04, 916.35 examples/s]3836 examples [00:04, 916.86 examples/s]3928 examples [00:04, 914.70 examples/s]4020 examples [00:04, 911.57 examples/s]4112 examples [00:04, 907.87 examples/s]4203 examples [00:04, 900.06 examples/s]4295 examples [00:04, 905.41 examples/s]4388 examples [00:04, 910.71 examples/s]4480 examples [00:04, 913.26 examples/s]4572 examples [00:05, 915.10 examples/s]4664 examples [00:05, 915.79 examples/s]4756 examples [00:05, 898.76 examples/s]4850 examples [00:05, 909.18 examples/s]4943 examples [00:05, 912.63 examples/s]5035 examples [00:05, 903.06 examples/s]5126 examples [00:05, 903.00 examples/s]5217 examples [00:05, 873.65 examples/s]5308 examples [00:05, 883.18 examples/s]5401 examples [00:05, 894.03 examples/s]5491 examples [00:06, 891.43 examples/s]5581 examples [00:06, 888.41 examples/s]5670 examples [00:06, 886.24 examples/s]5759 examples [00:06, 872.12 examples/s]5847 examples [00:06, 853.82 examples/s]5934 examples [00:06, 854.68 examples/s]6020 examples [00:06, 849.08 examples/s]6109 examples [00:06, 858.34 examples/s]6198 examples [00:06, 866.10 examples/s]6285 examples [00:07, 851.02 examples/s]6371 examples [00:07, 849.84 examples/s]6460 examples [00:07, 857.98 examples/s]6549 examples [00:07, 866.76 examples/s]6638 examples [00:07, 870.83 examples/s]6726 examples [00:07, 870.41 examples/s]6817 examples [00:07, 880.61 examples/s]6907 examples [00:07, 883.32 examples/s]6996 examples [00:07, 877.08 examples/s]7084 examples [00:07, 859.54 examples/s]7172 examples [00:08, 862.05 examples/s]7259 examples [00:08, 824.22 examples/s]7344 examples [00:08, 830.70 examples/s]7428 examples [00:08, 817.65 examples/s]7512 examples [00:08, 824.22 examples/s]7595 examples [00:08, 815.66 examples/s]7682 examples [00:08, 829.34 examples/s]7767 examples [00:08, 834.77 examples/s]7856 examples [00:08, 849.36 examples/s]7946 examples [00:08, 863.16 examples/s]8036 examples [00:09, 873.02 examples/s]8124 examples [00:09, 841.24 examples/s]8216 examples [00:09, 860.99 examples/s]8305 examples [00:09, 869.42 examples/s]8398 examples [00:09, 885.06 examples/s]8487 examples [00:09, 881.57 examples/s]8576 examples [00:09, 857.33 examples/s]8664 examples [00:09, 862.61 examples/s]8751 examples [00:09, 858.26 examples/s]8837 examples [00:09, 849.15 examples/s]8926 examples [00:10, 860.38 examples/s]9017 examples [00:10, 871.53 examples/s]9107 examples [00:10, 879.33 examples/s]9196 examples [00:10, 878.19 examples/s]9289 examples [00:10, 891.25 examples/s]9379 examples [00:10, 887.14 examples/s]9468 examples [00:10, 881.41 examples/s]9562 examples [00:10, 896.91 examples/s]9652 examples [00:10, 896.59 examples/s]9742 examples [00:11, 884.45 examples/s]9831 examples [00:11, 883.18 examples/s]9921 examples [00:11, 886.06 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 99%|█████████▊| 9873/10000 [00:00<00:00, 98727.03 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteXWNFEO/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteXWNFEO/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6b34116158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6b34116158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6b34116158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6b7faf60b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6b9f607f98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f6b34116158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f6b34116158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f6b34116158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6b32519320>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f6b325190f0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.47 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.47 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.47 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.30 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.30 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.62 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.09 file/s]2020-08-09 00:09:36.495399: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-09 00:09:36.499929: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-09 00:09:36.500100: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x563e2f150730 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-09 00:09:36.500118: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:08, 143954.93it/s] 68%|██████▊   | 6766592/9912422 [00:00<00:15, 205459.83it/s]9920512it [00:00, 41093265.19it/s]                           
0it [00:00, ?it/s]32768it [00:00, 481758.49it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 140066.89it/s]1654784it [00:00, 9683836.59it/s]                          
0it [00:00, ?it/s]8192it [00:00, 149747.61it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

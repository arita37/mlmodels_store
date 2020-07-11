
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1f926ca048> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1f926ca048> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1ffe2721e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1ffe2721e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f201c5b9ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f201c5b9ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fab5a0620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fab5a0620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fab5a0620> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 29%|██▉       | 2850816/9912422 [00:00<00:00, 28449041.29it/s]9920512it [00:00, 34691441.26it/s]                             
0it [00:00, ?it/s]32768it [00:00, 611369.65it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 462300.83it/s]1654784it [00:00, 11757784.38it/s]                         
0it [00:00, ?it/s]8192it [00:00, 168813.23it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1fa87c5668>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f925d2c18>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1fab5a0268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1fab5a0268> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1fab5a0268> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:54,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:53,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:52,  2.98 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:36,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:35,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:35,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:35,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:35,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:34,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:34,  4.17 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:24,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:24,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:24,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:24,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:15,  8.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:10, 11.16 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:06, 15.18 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 20.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 20.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 26.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 33.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.68 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 50.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.43 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 66.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 66.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 78.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 78.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 82.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 82.76 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 85.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.79 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.10s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.79 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.16s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.88 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.17s/ url]
0 examples [00:00, ? examples/s]2020-07-11 00:08:46.299417: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-11 00:08:46.314214: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-11 00:08:46.314488: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5615d702fd50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-11 00:08:46.314510: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.18 examples/s]94 examples [00:00,  3.11 examples/s]196 examples [00:00,  4.44 examples/s]285 examples [00:00,  6.33 examples/s]386 examples [00:00,  9.02 examples/s]485 examples [00:00, 12.83 examples/s]586 examples [00:01, 18.23 examples/s]692 examples [00:01, 25.85 examples/s]791 examples [00:01, 36.52 examples/s]891 examples [00:01, 51.36 examples/s]995 examples [00:01, 71.85 examples/s]1094 examples [00:01, 99.51 examples/s]1201 examples [00:01, 136.69 examples/s]1302 examples [00:01, 184.53 examples/s]1403 examples [00:01, 242.30 examples/s]1501 examples [00:01, 312.83 examples/s]1599 examples [00:02, 391.19 examples/s]1696 examples [00:02, 470.98 examples/s]1794 examples [00:02, 557.81 examples/s]1894 examples [00:02, 642.71 examples/s]1994 examples [00:02, 718.58 examples/s]2092 examples [00:02, 774.46 examples/s]2195 examples [00:02, 834.72 examples/s]2297 examples [00:02, 881.17 examples/s]2397 examples [00:02, 906.16 examples/s]2496 examples [00:03, 918.77 examples/s]2594 examples [00:03, 923.04 examples/s]2691 examples [00:03, 902.65 examples/s]2790 examples [00:03, 926.65 examples/s]2885 examples [00:03, 930.00 examples/s]2985 examples [00:03, 949.93 examples/s]3082 examples [00:03, 931.76 examples/s]3180 examples [00:03, 944.98 examples/s]3281 examples [00:03, 962.18 examples/s]3378 examples [00:03, 947.48 examples/s]3478 examples [00:04, 961.98 examples/s]3576 examples [00:04, 966.75 examples/s]3674 examples [00:04, 969.10 examples/s]3777 examples [00:04, 985.40 examples/s]3877 examples [00:04, 989.04 examples/s]3978 examples [00:04, 994.50 examples/s]4078 examples [00:04, 966.00 examples/s]4181 examples [00:04, 983.46 examples/s]4284 examples [00:04, 995.05 examples/s]4384 examples [00:04, 994.83 examples/s]4489 examples [00:05, 1008.44 examples/s]4596 examples [00:05, 1025.35 examples/s]4699 examples [00:05, 991.34 examples/s] 4811 examples [00:05, 1025.24 examples/s]4915 examples [00:05, 1010.50 examples/s]5017 examples [00:05, 994.09 examples/s] 5119 examples [00:05, 1000.94 examples/s]5220 examples [00:05, 994.64 examples/s] 5326 examples [00:05, 1012.89 examples/s]5428 examples [00:05, 1010.00 examples/s]5530 examples [00:06, 992.91 examples/s] 5632 examples [00:06, 999.16 examples/s]5733 examples [00:06, 983.14 examples/s]5840 examples [00:06, 1007.34 examples/s]5941 examples [00:06, 963.57 examples/s] 6038 examples [00:06, 928.72 examples/s]6134 examples [00:06, 936.56 examples/s]6229 examples [00:06, 905.74 examples/s]6321 examples [00:06, 869.56 examples/s]6409 examples [00:07, 872.39 examples/s]6500 examples [00:07, 883.30 examples/s]6598 examples [00:07, 907.45 examples/s]6693 examples [00:07, 918.82 examples/s]6792 examples [00:07, 937.01 examples/s]6890 examples [00:07, 947.75 examples/s]6990 examples [00:07, 961.52 examples/s]7090 examples [00:07, 970.28 examples/s]7189 examples [00:07, 975.60 examples/s]7287 examples [00:07, 961.03 examples/s]7384 examples [00:08, 940.65 examples/s]7479 examples [00:08, 933.53 examples/s]7573 examples [00:08, 920.96 examples/s]7673 examples [00:08, 941.12 examples/s]7768 examples [00:08, 936.87 examples/s]7862 examples [00:08, 927.22 examples/s]7966 examples [00:08, 956.01 examples/s]8062 examples [00:08, 948.17 examples/s]8165 examples [00:08, 969.01 examples/s]8263 examples [00:09, 954.07 examples/s]8359 examples [00:09, 937.03 examples/s]8453 examples [00:09, 896.38 examples/s]8544 examples [00:09, 875.40 examples/s]8645 examples [00:09, 909.93 examples/s]8739 examples [00:09, 918.23 examples/s]8835 examples [00:09, 929.10 examples/s]8929 examples [00:09, 929.96 examples/s]9024 examples [00:09, 934.20 examples/s]9126 examples [00:09, 957.25 examples/s]9222 examples [00:10, 896.44 examples/s]9317 examples [00:10, 902.79 examples/s]9409 examples [00:10, 907.48 examples/s]9501 examples [00:10, 901.93 examples/s]9600 examples [00:10, 924.47 examples/s]9693 examples [00:10, 898.67 examples/s]9785 examples [00:10, 903.79 examples/s]9884 examples [00:10, 926.18 examples/s]9982 examples [00:10, 940.86 examples/s]10077 examples [00:11, 883.82 examples/s]10176 examples [00:11, 911.82 examples/s]10281 examples [00:11, 948.28 examples/s]10380 examples [00:11, 958.85 examples/s]10483 examples [00:11, 978.14 examples/s]10582 examples [00:11, 975.71 examples/s]10680 examples [00:11, 939.92 examples/s]10787 examples [00:11, 973.52 examples/s]10886 examples [00:11, 922.73 examples/s]10981 examples [00:11, 930.39 examples/s]11075 examples [00:12, 910.56 examples/s]11167 examples [00:12, 900.93 examples/s]11258 examples [00:12, 898.28 examples/s]11349 examples [00:12, 877.68 examples/s]11448 examples [00:12, 907.94 examples/s]11548 examples [00:12, 932.77 examples/s]11647 examples [00:12, 947.61 examples/s]11748 examples [00:12, 962.09 examples/s]11845 examples [00:12, 955.28 examples/s]11943 examples [00:12, 959.52 examples/s]12040 examples [00:13, 960.39 examples/s]12137 examples [00:13, 931.91 examples/s]12231 examples [00:13, 934.03 examples/s]12336 examples [00:13, 964.34 examples/s]12440 examples [00:13, 984.37 examples/s]12539 examples [00:13, 985.89 examples/s]12643 examples [00:13, 999.34 examples/s]12745 examples [00:13, 1004.57 examples/s]12852 examples [00:13, 1022.37 examples/s]12960 examples [00:13, 1036.34 examples/s]13064 examples [00:14, 1018.47 examples/s]13167 examples [00:14, 1007.04 examples/s]13268 examples [00:14, 1007.90 examples/s]13377 examples [00:14, 1030.60 examples/s]13481 examples [00:14, 1028.87 examples/s]13585 examples [00:14, 953.59 examples/s] 13694 examples [00:14, 990.41 examples/s]13795 examples [00:14, 987.01 examples/s]13902 examples [00:14, 1009.36 examples/s]14007 examples [00:15, 1018.75 examples/s]14110 examples [00:15, 1010.84 examples/s]14212 examples [00:15, 984.40 examples/s] 14311 examples [00:15, 984.49 examples/s]14415 examples [00:15, 996.78 examples/s]14515 examples [00:15, 992.87 examples/s]14615 examples [00:15, 983.34 examples/s]14714 examples [00:15, 970.42 examples/s]14812 examples [00:15, 945.07 examples/s]14910 examples [00:15, 953.04 examples/s]15007 examples [00:16, 956.35 examples/s]15105 examples [00:16, 961.68 examples/s]15203 examples [00:16, 965.71 examples/s]15300 examples [00:16, 966.73 examples/s]15398 examples [00:16, 970.07 examples/s]15497 examples [00:16, 975.16 examples/s]15595 examples [00:16, 950.75 examples/s]15693 examples [00:16, 958.19 examples/s]15789 examples [00:16, 950.99 examples/s]15885 examples [00:16, 951.35 examples/s]15987 examples [00:17, 970.54 examples/s]16085 examples [00:17, 968.87 examples/s]16182 examples [00:17, 967.58 examples/s]16279 examples [00:17, 958.68 examples/s]16382 examples [00:17, 976.42 examples/s]16480 examples [00:17, 956.68 examples/s]16576 examples [00:17, 952.25 examples/s]16680 examples [00:17, 976.77 examples/s]16778 examples [00:17, 950.96 examples/s]16874 examples [00:18, 949.48 examples/s]16970 examples [00:18, 944.58 examples/s]17065 examples [00:18, 933.45 examples/s]17159 examples [00:18, 928.02 examples/s]17260 examples [00:18, 950.64 examples/s]17361 examples [00:18, 965.40 examples/s]17461 examples [00:18, 974.20 examples/s]17559 examples [00:18, 974.20 examples/s]17665 examples [00:18, 995.52 examples/s]17765 examples [00:18, 996.22 examples/s]17865 examples [00:19, 993.11 examples/s]17965 examples [00:19, 979.72 examples/s]18064 examples [00:19, 948.37 examples/s]18160 examples [00:19, 941.34 examples/s]18257 examples [00:19, 949.70 examples/s]18359 examples [00:19, 969.69 examples/s]18457 examples [00:19, 970.95 examples/s]18555 examples [00:19, 971.94 examples/s]18653 examples [00:19, 973.86 examples/s]18751 examples [00:19, 975.64 examples/s]18849 examples [00:20, 974.19 examples/s]18949 examples [00:20, 981.76 examples/s]19048 examples [00:20, 969.68 examples/s]19151 examples [00:20, 984.69 examples/s]19261 examples [00:20, 1014.02 examples/s]19363 examples [00:20, 993.54 examples/s] 19468 examples [00:20, 1008.39 examples/s]19570 examples [00:20, 996.51 examples/s] 19675 examples [00:20, 1011.80 examples/s]19777 examples [00:20, 1012.99 examples/s]19879 examples [00:21, 997.47 examples/s] 19979 examples [00:21, 980.70 examples/s]20078 examples [00:21, 891.03 examples/s]20185 examples [00:21, 937.62 examples/s]20292 examples [00:21, 973.62 examples/s]20392 examples [00:21, 969.95 examples/s]20491 examples [00:21, 946.75 examples/s]20587 examples [00:21, 888.28 examples/s]20678 examples [00:21, 894.27 examples/s]20776 examples [00:22, 916.04 examples/s]20875 examples [00:22, 935.75 examples/s]20970 examples [00:22, 924.55 examples/s]21065 examples [00:22, 930.43 examples/s]21161 examples [00:22, 937.58 examples/s]21264 examples [00:22, 962.68 examples/s]21364 examples [00:22, 971.39 examples/s]21462 examples [00:22, 956.58 examples/s]21558 examples [00:22, 898.65 examples/s]21659 examples [00:22, 926.68 examples/s]21753 examples [00:23, 864.66 examples/s]21841 examples [00:23, 844.15 examples/s]21931 examples [00:23, 859.72 examples/s]22021 examples [00:23, 870.49 examples/s]22119 examples [00:23, 899.27 examples/s]22210 examples [00:23, 883.51 examples/s]22300 examples [00:23, 885.03 examples/s]22389 examples [00:23, 840.71 examples/s]22481 examples [00:23, 862.27 examples/s]22583 examples [00:24, 902.20 examples/s]22686 examples [00:24, 935.12 examples/s]22781 examples [00:24, 933.78 examples/s]22876 examples [00:24, 909.66 examples/s]22975 examples [00:24, 930.28 examples/s]23069 examples [00:24, 907.82 examples/s]23166 examples [00:24, 925.48 examples/s]23261 examples [00:24, 930.70 examples/s]23356 examples [00:24, 934.75 examples/s]23453 examples [00:24, 943.96 examples/s]23554 examples [00:25, 960.50 examples/s]23654 examples [00:25, 971.54 examples/s]23752 examples [00:25, 962.72 examples/s]23851 examples [00:25, 970.65 examples/s]23949 examples [00:25, 967.85 examples/s]24046 examples [00:25, 966.26 examples/s]24143 examples [00:25, 966.45 examples/s]24240 examples [00:25, 949.00 examples/s]24335 examples [00:25, 947.53 examples/s]24436 examples [00:26, 964.50 examples/s]24534 examples [00:26, 968.82 examples/s]24632 examples [00:26, 971.70 examples/s]24730 examples [00:26, 952.61 examples/s]24826 examples [00:26, 943.36 examples/s]24922 examples [00:26, 947.50 examples/s]25017 examples [00:26, 941.15 examples/s]25118 examples [00:26, 959.81 examples/s]25215 examples [00:26, 917.32 examples/s]25309 examples [00:26, 923.69 examples/s]25405 examples [00:27, 932.80 examples/s]25499 examples [00:27, 931.52 examples/s]25595 examples [00:27, 939.35 examples/s]25692 examples [00:27, 947.67 examples/s]25787 examples [00:27, 922.19 examples/s]25880 examples [00:27, 889.54 examples/s]25973 examples [00:27, 899.70 examples/s]26079 examples [00:27, 940.26 examples/s]26180 examples [00:27, 957.74 examples/s]26284 examples [00:27, 978.74 examples/s]26390 examples [00:28, 1000.23 examples/s]26491 examples [00:28, 970.92 examples/s] 26589 examples [00:28, 941.46 examples/s]26684 examples [00:28, 929.75 examples/s]26778 examples [00:28, 919.06 examples/s]26875 examples [00:28, 930.57 examples/s]26969 examples [00:28, 931.80 examples/s]27064 examples [00:28, 937.08 examples/s]27158 examples [00:28, 924.87 examples/s]27251 examples [00:29, 910.68 examples/s]27351 examples [00:29, 935.39 examples/s]27447 examples [00:29, 941.59 examples/s]27547 examples [00:29, 958.34 examples/s]27645 examples [00:29, 964.16 examples/s]27745 examples [00:29, 974.23 examples/s]27844 examples [00:29, 978.04 examples/s]27951 examples [00:29, 1001.40 examples/s]28052 examples [00:29, 981.04 examples/s] 28151 examples [00:29, 977.16 examples/s]28250 examples [00:30, 979.83 examples/s]28349 examples [00:30, 942.87 examples/s]28444 examples [00:30, 911.41 examples/s]28537 examples [00:30, 915.19 examples/s]28629 examples [00:30, 892.98 examples/s]28719 examples [00:30, 892.89 examples/s]28819 examples [00:30, 920.35 examples/s]28927 examples [00:30, 960.95 examples/s]29024 examples [00:30, 958.26 examples/s]29129 examples [00:30, 981.84 examples/s]29231 examples [00:31, 992.97 examples/s]29334 examples [00:31, 1002.72 examples/s]29441 examples [00:31, 1020.42 examples/s]29544 examples [00:31, 1001.98 examples/s]29645 examples [00:31, 951.23 examples/s] 29741 examples [00:31, 917.25 examples/s]29834 examples [00:31, 899.85 examples/s]29925 examples [00:31, 879.76 examples/s]30014 examples [00:31, 830.72 examples/s]30109 examples [00:32, 861.59 examples/s]30212 examples [00:32, 904.33 examples/s]30309 examples [00:32, 921.28 examples/s]30409 examples [00:32, 942.35 examples/s]30505 examples [00:32, 937.51 examples/s]30608 examples [00:32, 962.76 examples/s]30705 examples [00:32, 925.68 examples/s]30803 examples [00:32, 940.86 examples/s]30910 examples [00:32, 974.97 examples/s]31012 examples [00:32, 985.60 examples/s]31117 examples [00:33, 1001.91 examples/s]31218 examples [00:33, 984.71 examples/s] 31317 examples [00:33, 977.09 examples/s]31425 examples [00:33, 1005.18 examples/s]31526 examples [00:33, 1003.24 examples/s]31627 examples [00:33, 981.15 examples/s] 31730 examples [00:33, 994.45 examples/s]31832 examples [00:33, 1001.19 examples/s]31935 examples [00:33, 1009.37 examples/s]32037 examples [00:33, 1000.26 examples/s]32138 examples [00:34, 994.06 examples/s] 32238 examples [00:34, 986.97 examples/s]32337 examples [00:34, 962.01 examples/s]32434 examples [00:34, 951.37 examples/s]32530 examples [00:34, 944.89 examples/s]32630 examples [00:34, 958.42 examples/s]32729 examples [00:34, 967.32 examples/s]32827 examples [00:34, 970.88 examples/s]32925 examples [00:34, 964.14 examples/s]33022 examples [00:35, 947.62 examples/s]33117 examples [00:35, 942.41 examples/s]33212 examples [00:35, 936.13 examples/s]33306 examples [00:35, 919.65 examples/s]33399 examples [00:35, 907.98 examples/s]33490 examples [00:35, 905.94 examples/s]33581 examples [00:35, 885.28 examples/s]33679 examples [00:35, 910.69 examples/s]33779 examples [00:35, 935.66 examples/s]33882 examples [00:35, 958.05 examples/s]33979 examples [00:36, 955.77 examples/s]34084 examples [00:36, 980.23 examples/s]34183 examples [00:36, 979.56 examples/s]34282 examples [00:36, 972.87 examples/s]34380 examples [00:36, 962.42 examples/s]34477 examples [00:36, 931.56 examples/s]34571 examples [00:36, 888.08 examples/s]34662 examples [00:36, 893.57 examples/s]34758 examples [00:36, 912.33 examples/s]34859 examples [00:36, 937.77 examples/s]34957 examples [00:37, 949.48 examples/s]35053 examples [00:37, 938.00 examples/s]35148 examples [00:37, 919.97 examples/s]35248 examples [00:37, 941.86 examples/s]35351 examples [00:37, 964.30 examples/s]35448 examples [00:37, 949.33 examples/s]35544 examples [00:37, 941.23 examples/s]35640 examples [00:37, 946.46 examples/s]35737 examples [00:37, 951.39 examples/s]35836 examples [00:38, 960.58 examples/s]35933 examples [00:38, 951.43 examples/s]36029 examples [00:38, 936.56 examples/s]36130 examples [00:38, 955.55 examples/s]36232 examples [00:38, 973.56 examples/s]36331 examples [00:38, 976.48 examples/s]36429 examples [00:38, 963.37 examples/s]36527 examples [00:38, 966.18 examples/s]36625 examples [00:38, 969.31 examples/s]36731 examples [00:38, 994.73 examples/s]36838 examples [00:39, 1015.21 examples/s]36940 examples [00:39, 1008.31 examples/s]37043 examples [00:39, 1013.00 examples/s]37146 examples [00:39, 1016.13 examples/s]37248 examples [00:39, 1010.00 examples/s]37350 examples [00:39, 981.79 examples/s] 37449 examples [00:39, 905.73 examples/s]37550 examples [00:39, 933.05 examples/s]37652 examples [00:39, 955.56 examples/s]37754 examples [00:39, 971.88 examples/s]37853 examples [00:40, 974.92 examples/s]37951 examples [00:40, 964.14 examples/s]38056 examples [00:40, 986.18 examples/s]38156 examples [00:40, 985.86 examples/s]38255 examples [00:40, 986.08 examples/s]38355 examples [00:40, 988.03 examples/s]38454 examples [00:40, 977.21 examples/s]38563 examples [00:40, 1006.92 examples/s]38669 examples [00:40, 1021.94 examples/s]38772 examples [00:41, 1000.36 examples/s]38873 examples [00:41, 945.77 examples/s] 38969 examples [00:41, 918.27 examples/s]39067 examples [00:41, 934.50 examples/s]39167 examples [00:41, 949.99 examples/s]39268 examples [00:41, 966.08 examples/s]39366 examples [00:41, 966.84 examples/s]39463 examples [00:41, 923.90 examples/s]39556 examples [00:41, 917.44 examples/s]39653 examples [00:41, 932.24 examples/s]39747 examples [00:42, 900.21 examples/s]39839 examples [00:42, 905.60 examples/s]39936 examples [00:42, 921.59 examples/s]40029 examples [00:42, 892.56 examples/s]40131 examples [00:42, 925.00 examples/s]40229 examples [00:42, 940.50 examples/s]40324 examples [00:42, 912.62 examples/s]40416 examples [00:42, 903.87 examples/s]40507 examples [00:42, 892.11 examples/s]40606 examples [00:43, 917.38 examples/s]40701 examples [00:43, 926.01 examples/s]40794 examples [00:43, 915.90 examples/s]40892 examples [00:43, 930.85 examples/s]40988 examples [00:43, 937.95 examples/s]41089 examples [00:43, 956.75 examples/s]41185 examples [00:43, 950.45 examples/s]41281 examples [00:43, 939.82 examples/s]41387 examples [00:43, 971.45 examples/s]41485 examples [00:43, 971.97 examples/s]41586 examples [00:44, 980.99 examples/s]41685 examples [00:44, 977.05 examples/s]41783 examples [00:44, 944.45 examples/s]41883 examples [00:44, 957.20 examples/s]41981 examples [00:44, 963.38 examples/s]42078 examples [00:44, 956.71 examples/s]42183 examples [00:44, 980.87 examples/s]42282 examples [00:44, 946.45 examples/s]42378 examples [00:44, 943.90 examples/s]42473 examples [00:44, 939.77 examples/s]42572 examples [00:45, 952.83 examples/s]42668 examples [00:45, 940.84 examples/s]42763 examples [00:45, 891.33 examples/s]42861 examples [00:45, 913.72 examples/s]42957 examples [00:45, 925.21 examples/s]43053 examples [00:45, 934.37 examples/s]43155 examples [00:45, 955.67 examples/s]43251 examples [00:45, 905.73 examples/s]43348 examples [00:45, 923.12 examples/s]43445 examples [00:46, 935.15 examples/s]43544 examples [00:46, 950.31 examples/s]43644 examples [00:46, 963.02 examples/s]43742 examples [00:46, 967.22 examples/s]43844 examples [00:46, 980.58 examples/s]43947 examples [00:46, 992.08 examples/s]44049 examples [00:46, 998.63 examples/s]44149 examples [00:46, 975.92 examples/s]44247 examples [00:46, 907.01 examples/s]44341 examples [00:46, 916.15 examples/s]44441 examples [00:47, 938.90 examples/s]44536 examples [00:47, 939.06 examples/s]44631 examples [00:47, 914.65 examples/s]44728 examples [00:47, 930.42 examples/s]44822 examples [00:47, 928.03 examples/s]44922 examples [00:47, 946.82 examples/s]45018 examples [00:47, 949.25 examples/s]45114 examples [00:47, 950.78 examples/s]45213 examples [00:47, 962.07 examples/s]45311 examples [00:47, 966.27 examples/s]45410 examples [00:48, 973.19 examples/s]45508 examples [00:48, 937.71 examples/s]45603 examples [00:48, 915.94 examples/s]45700 examples [00:48, 930.05 examples/s]45794 examples [00:48, 922.68 examples/s]45893 examples [00:48, 941.78 examples/s]45994 examples [00:48, 960.26 examples/s]46091 examples [00:48, 935.35 examples/s]46189 examples [00:48, 946.60 examples/s]46285 examples [00:48, 950.15 examples/s]46390 examples [00:49, 977.48 examples/s]46492 examples [00:49, 987.34 examples/s]46591 examples [00:49, 984.84 examples/s]46693 examples [00:49, 995.10 examples/s]46794 examples [00:49, 998.78 examples/s]46897 examples [00:49, 1006.12 examples/s]47000 examples [00:49, 1012.21 examples/s]47102 examples [00:49, 996.40 examples/s] 47205 examples [00:49, 1003.68 examples/s]47306 examples [00:50, 972.90 examples/s] 47410 examples [00:50, 991.85 examples/s]47510 examples [00:50, 992.15 examples/s]47610 examples [00:50, 979.12 examples/s]47717 examples [00:50, 1003.34 examples/s]47820 examples [00:50, 1008.76 examples/s]47926 examples [00:50, 1022.95 examples/s]48029 examples [00:50, 987.16 examples/s] 48129 examples [00:50, 963.97 examples/s]48227 examples [00:50, 966.14 examples/s]48324 examples [00:51, 960.76 examples/s]48426 examples [00:51, 975.13 examples/s]48524 examples [00:51, 956.57 examples/s]48620 examples [00:51, 944.20 examples/s]48724 examples [00:51, 968.99 examples/s]48822 examples [00:51, 945.48 examples/s]48917 examples [00:51, 927.74 examples/s]49016 examples [00:51, 943.22 examples/s]49111 examples [00:51, 929.45 examples/s]49213 examples [00:51, 954.31 examples/s]49309 examples [00:52, 954.49 examples/s]49408 examples [00:52, 964.60 examples/s]49505 examples [00:52, 957.45 examples/s]49601 examples [00:52, 916.84 examples/s]49702 examples [00:52, 941.86 examples/s]49799 examples [00:52, 949.58 examples/s]49896 examples [00:52, 953.34 examples/s]49992 examples [00:52, 942.05 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 15%|█▍        | 7351/50000 [00:00<00:00, 73504.64 examples/s] 39%|███▉      | 19561/50000 [00:00<00:00, 83469.61 examples/s] 62%|██████▏   | 30832/50000 [00:00<00:00, 90512.56 examples/s] 84%|████████▎ | 41755/50000 [00:00<00:00, 95416.63 examples/s]                                                               0 examples [00:00, ? examples/s]73 examples [00:00, 727.93 examples/s]169 examples [00:00, 784.33 examples/s]272 examples [00:00, 844.15 examples/s]372 examples [00:00, 885.48 examples/s]471 examples [00:00, 912.83 examples/s]569 examples [00:00, 930.73 examples/s]667 examples [00:00, 944.17 examples/s]767 examples [00:00, 959.02 examples/s]865 examples [00:00, 963.15 examples/s]961 examples [00:01, 961.67 examples/s]1056 examples [00:01, 946.56 examples/s]1154 examples [00:01, 955.98 examples/s]1256 examples [00:01, 972.78 examples/s]1355 examples [00:01, 975.21 examples/s]1453 examples [00:01, 955.10 examples/s]1549 examples [00:01, 939.03 examples/s]1652 examples [00:01, 961.44 examples/s]1759 examples [00:01, 990.66 examples/s]1861 examples [00:01, 998.92 examples/s]1965 examples [00:02, 1006.47 examples/s]2066 examples [00:02, 1005.06 examples/s]2167 examples [00:02, 956.30 examples/s] 2266 examples [00:02, 966.08 examples/s]2365 examples [00:02, 972.20 examples/s]2463 examples [00:02, 962.45 examples/s]2560 examples [00:02, 955.56 examples/s]2660 examples [00:02, 967.89 examples/s]2761 examples [00:02, 979.38 examples/s]2863 examples [00:02, 989.22 examples/s]2963 examples [00:03, 967.43 examples/s]3060 examples [00:03, 928.42 examples/s]3156 examples [00:03, 937.22 examples/s]3251 examples [00:03, 926.59 examples/s]3344 examples [00:03, 920.90 examples/s]3437 examples [00:03, 920.53 examples/s]3530 examples [00:03, 921.14 examples/s]3623 examples [00:03, 910.21 examples/s]3721 examples [00:03, 928.11 examples/s]3821 examples [00:03, 944.95 examples/s]3917 examples [00:04, 948.56 examples/s]4020 examples [00:04, 968.73 examples/s]4118 examples [00:04, 958.60 examples/s]4215 examples [00:04, 948.23 examples/s]4310 examples [00:04, 921.64 examples/s]4403 examples [00:04, 897.37 examples/s]4503 examples [00:04, 924.41 examples/s]4599 examples [00:04, 933.15 examples/s]4694 examples [00:04, 937.45 examples/s]4801 examples [00:05, 972.05 examples/s]4899 examples [00:05, 973.05 examples/s]4997 examples [00:05, 975.09 examples/s]5101 examples [00:05, 992.74 examples/s]5210 examples [00:05, 1020.00 examples/s]5315 examples [00:05, 1028.53 examples/s]5419 examples [00:05, 994.99 examples/s] 5519 examples [00:05, 978.77 examples/s]5618 examples [00:05, 921.85 examples/s]5712 examples [00:05, 925.49 examples/s]5806 examples [00:06, 878.31 examples/s]5901 examples [00:06, 896.88 examples/s]6001 examples [00:06, 924.80 examples/s]6095 examples [00:06, 891.73 examples/s]6188 examples [00:06, 902.32 examples/s]6283 examples [00:06, 915.19 examples/s]6375 examples [00:06, 911.21 examples/s]6475 examples [00:06, 934.32 examples/s]6569 examples [00:06, 913.30 examples/s]6670 examples [00:07, 939.70 examples/s]6772 examples [00:07, 961.00 examples/s]6870 examples [00:07, 965.70 examples/s]6971 examples [00:07, 976.93 examples/s]7076 examples [00:07, 996.18 examples/s]7176 examples [00:07, 986.41 examples/s]7275 examples [00:07, 983.37 examples/s]7374 examples [00:07, 949.62 examples/s]7476 examples [00:07, 969.24 examples/s]7574 examples [00:07, 945.20 examples/s]7669 examples [00:08, 930.39 examples/s]7769 examples [00:08, 948.43 examples/s]7869 examples [00:08, 961.31 examples/s]7974 examples [00:08, 984.55 examples/s]8073 examples [00:08, 956.67 examples/s]8170 examples [00:08, 932.12 examples/s]8264 examples [00:08, 927.50 examples/s]8358 examples [00:08, 913.52 examples/s]8450 examples [00:08, 903.75 examples/s]8543 examples [00:08, 909.60 examples/s]8638 examples [00:09, 919.42 examples/s]8731 examples [00:09, 918.42 examples/s]8823 examples [00:09, 916.26 examples/s]8921 examples [00:09, 933.41 examples/s]9015 examples [00:09, 932.11 examples/s]9109 examples [00:09, 931.93 examples/s]9207 examples [00:09, 942.13 examples/s]9302 examples [00:09, 944.23 examples/s]9397 examples [00:09, 937.59 examples/s]9491 examples [00:09, 936.94 examples/s]9593 examples [00:10, 960.03 examples/s]9690 examples [00:10, 915.85 examples/s]9783 examples [00:10, 911.61 examples/s]9884 examples [00:10, 938.82 examples/s]9988 examples [00:10, 965.29 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW985O1/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteW985O1/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fab5a0620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fab5a0620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fab5a0620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f30a1ef98>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f30a1eb38>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1fab5a0620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1fab5a0620> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1fab5a0620> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f30979588>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1f925d22e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1f233bf400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1f233bf400> 

  function with postional parmater data_info <function split_train_valid at 0x7f1f233bf400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1f233bf510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1f233bf510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1f233bf510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.1
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.1) (2.3.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.7.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.2)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.19.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (4.47.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (0.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.0.2)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.24.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.0.3)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.7.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.0.4)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2.10)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (1.25.9)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.1) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.1-py3-none-any.whl size=12047105 sha256=fb953b658dca7161213e0743dad86d6b0039a8846cf7b23787765be2f782c75e
  Stored in directory: /tmp/pip-ephem-wheel-cache-s9bo13wz/wheels/10/6f/a6/ddd8204ceecdedddea923f8514e13afb0c1f0f556d2c9c3da0
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.3.1
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:55:32, 10.9kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:34:43, 15.4kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:57:29, 21.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:40:42, 31.2kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:21:40, 44.5kB/s].vector_cache/glove.6B.zip:   1%|          | 9.32M/862M [00:01<3:43:47, 63.5kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:01<2:35:43, 90.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.9M/862M [00:01<1:48:45, 129kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 23.3M/862M [00:01<1:15:44, 185kB/s].vector_cache/glove.6B.zip:   3%|▎         | 28.0M/862M [00:01<52:48, 263kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:01<36:52, 375kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.0M/862M [00:02<25:42, 534kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.1M/862M [00:02<18:04, 757kB/s].vector_cache/glove.6B.zip:   5%|▌         | 46.9M/862M [00:02<12:37, 1.08MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.9M/862M [00:02<08:56, 1.51MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:02<06:24, 2.11MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:04<07:01, 1.91MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<08:31, 1.58MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<06:53, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.6M/862M [00:04<05:02, 2.66MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:06<09:20, 1.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<07:50, 1.70MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<05:57, 2.24MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.2M/862M [00:06<04:22, 3.05MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<12:18, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.1M/862M [00:08<11:17, 1.18MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<08:34, 1.55MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:08<06:07, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:10<12:51:55, 17.1kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.4M/862M [00:10<9:01:26, 24.4kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:10<6:18:34, 34.9kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<4:27:21, 49.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<3:09:45, 69.4kB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.1M/862M [00:12<2:13:22, 98.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.2M/862M [00:14<1:35:07, 138kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<1:07:53, 193kB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.2M/862M [00:14<47:43, 274kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:16<36:23, 358kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<28:05, 464kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.3M/862M [00:16<20:12, 644kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.8M/862M [00:16<14:15, 910kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<19:33, 663kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.9M/862M [00:18<15:02, 862kB/s].vector_cache/glove.6B.zip:  10%|█         | 86.4M/862M [00:18<10:49, 1.19MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<10:35, 1.22MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<10:08, 1.27MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<07:40, 1.68MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.0M/862M [00:20<05:30, 2.33MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:22<13:27, 953kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.1M/862M [00:22<10:44, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.6M/862M [00:22<07:46, 1.65MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<08:26, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.0M/862M [00:24<08:28, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<06:34, 1.94MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:24<04:43, 2.69MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<12:20:43, 17.1kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<8:39:30, 24.4kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<6:03:12, 34.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<4:16:30, 49.2kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<3:02:06, 69.3kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<2:07:58, 98.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<1:29:21, 140kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<13:02:16, 16.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<9:08:36, 22.9kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<6:23:34, 32.6kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<4:30:36, 46.1kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:32<3:11:56, 65.0kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<2:14:46, 92.5kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<1:34:16, 132kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<1:11:16, 174kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<51:09, 243kB/s]  .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<36:03, 343kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<28:03, 440kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<22:08, 558kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<16:06, 766kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<13:13, 929kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<11:44, 1.05MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:46, 1.40MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<06:14, 1.96MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<46:44, 261kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<33:57, 359kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<24:02, 507kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<19:36, 619kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<16:11, 750kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<11:55, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<10:16, 1.17MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<08:25, 1.43MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:43<06:11, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<07:08, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<07:27, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<05:44, 2.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<04:08, 2.88MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<11:06, 1.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<08:58, 1.33MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:47<06:31, 1.82MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<07:20, 1.61MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:20, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:49<04:44, 2.50MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<06:05, 1.93MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<05:16, 2.24MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<03:59, 2.95MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<05:33, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<06:15, 1.87MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<04:52, 2.40MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:53<03:33, 3.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:56, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<07:26, 1.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:55<05:29, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:33, 1.77MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<06:55, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<05:26, 2.12MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:57<03:56, 2.92MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:36:51, 119kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<1:08:56, 167kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [00:59<48:26, 237kB/s]  .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<36:29, 314kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<27:51, 411kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<19:57, 573kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:01<14:04, 810kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<17:25, 654kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<13:21, 852kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<09:34, 1.19MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<09:20, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<08:49, 1.28MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<06:44, 1.68MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<06:31, 1.72MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:41, 1.97MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:16, 2.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<05:36, 1.99MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:10, 1.81MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<04:50, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:09<03:31, 3.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<08:42, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:12, 1.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:19, 2.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:18, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:37, 1.66MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:07, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<03:49, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:42, 1.92MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<05:07, 2.14MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<03:49, 2.86MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<05:13, 2.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<04:46, 2.28MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:36, 3.01MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<05:04, 2.14MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<04:27, 2.43MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<03:19, 3.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:19<02:28, 4.34MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<45:10, 238kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<33:47, 319kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<24:09, 445kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<18:34, 576kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<14:04, 760kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<10:03, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<09:31, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<07:33, 1.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:32, 1.91MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<06:22, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<06:35, 1.60MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:27<05:09, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:27<03:42, 2.84MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<20:37, 509kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<15:30, 676kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<11:05, 943kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<10:10, 1.02MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<09:13, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<06:53, 1.51MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:59, 2.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<07:10, 1.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:05, 1.70MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<04:28, 2.31MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:33, 1.85MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:57, 1.73MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:35<04:41, 2.19MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:55, 2.07MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:18, 2.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<03:22, 3.02MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<02:29, 4.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<1:02:16, 163kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<45:37, 222kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<32:24, 313kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<24:12, 416kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<17:58, 560kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<12:48, 784kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<11:16, 888kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<09:58, 1.00MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<07:23, 1.35MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<05:21, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<06:48, 1.46MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:47, 1.71MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:17, 2.31MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:19, 1.86MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:42, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:29, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:43, 2.08MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<04:19, 2.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<03:15, 3.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:33, 2.13MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<04:10, 2.33MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<03:09, 3.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:29, 2.15MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:06, 1.89MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<04:01, 2.40MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<02:54, 3.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:53, 970kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<07:55, 1.21MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<05:44, 1.67MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:14, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:17, 1.52MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:48, 1.98MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:28, 2.72MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<07:08, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:58, 1.59MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:24, 2.14MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:16, 1.78MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:35, 1.68MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:18, 2.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:08, 2.97MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:14, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<05:08, 1.81MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<03:48, 2.44MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:02<02:47, 3.33MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<19:28, 476kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<14:32, 637kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:04<10:20, 892kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<09:24, 977kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<07:30, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:06<05:28, 1.67MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<05:58, 1.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<06:01, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<04:40, 1.95MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:43, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:13, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:08, 2.87MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<04:18, 2.09MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<03:55, 2.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<02:57, 3.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:10, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<04:43, 1.89MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<03:45, 2.37MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:43, 3.25MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<1:05:39, 135kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:16<46:49, 189kB/s]  .vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<32:52, 268kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<24:59, 351kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<19:15, 456kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<13:50, 633kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:18<09:44, 895kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<12:48, 680kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<09:51, 883kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<07:03, 1.23MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:58, 1.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<06:37, 1.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<05:01, 1.72MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:22<03:36, 2.38MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<1:04:20, 133kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<45:51, 187kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<32:11, 265kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<24:27, 348kB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<18:49, 452kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<13:34, 625kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<10:49, 780kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<08:25, 1.00MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<06:05, 1.38MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<06:12, 1.35MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:11, 1.61MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<03:49, 2.18MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:38, 1.79MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:55, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<03:52, 2.14MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<04:01, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<03:38, 2.25MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<02:45, 2.98MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<03:50, 2.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:16, 1.91MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:21, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<02:26, 3.32MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<07:11, 1.13MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:51, 1.38MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:15, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:52, 1.65MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<05:01, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:55, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<04:01, 1.98MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:36, 2.20MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:43, 2.91MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:45, 2.10MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:25, 2.30MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:35, 3.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:39, 2.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<03:20, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:31, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<03:36, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:18, 2.35MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:29, 3.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:34, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<04:02, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:09, 2.43MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<02:20, 3.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:59, 1.90MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<03:34, 2.13MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:41, 2.82MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:39, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:04, 1.85MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<03:10, 2.37MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:21, 3.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:53, 1.92MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:21, 2.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:31, 2.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:55<01:52, 3.96MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<45:25, 163kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<33:17, 222kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<23:37, 313kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:57<16:32, 444kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<16:01, 458kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<11:56, 614kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<08:31, 857kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<07:38, 952kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<06:03, 1.20MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<04:24, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:46, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:47, 1.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:42, 1.93MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:03<02:39, 2.68MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<09:43, 733kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:31, 946kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:26, 1.31MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<05:26, 1.30MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:07<05:14, 1.35MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:00, 1.76MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:55, 1.78MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<04:10, 1.67MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:13, 2.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:20, 2.96MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:32, 1.52MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:53, 1.78MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:53, 2.38MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:37, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:58, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<03:04, 2.23MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<02:14, 3.04MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<04:39, 1.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<03:57, 1.71MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<02:54, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:36, 1.86MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<03:53, 1.73MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:03, 2.20MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<03:12, 2.08MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<02:54, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:18<02:10, 3.05MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:04, 2.15MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:29, 1.88MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<02:43, 2.42MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<01:58, 3.31MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<06:04, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<04:55, 1.32MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<03:35, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:01, 1.60MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:10, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<03:11, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:18, 2.76MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<04:38, 1.37MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<03:48, 1.68MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<02:49, 2.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:25, 1.84MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<03:43, 1.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<02:52, 2.19MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:28<02:04, 3.00MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:58, 1.25MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<04:07, 1.51MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:01, 2.05MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:33, 1.73MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<03:44, 1.65MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:55, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:01, 2.02MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:43, 2.24MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:03, 2.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<02:51, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<03:09, 1.91MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:28, 2.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<01:47, 3.34MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<05:24, 1.10MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<04:23, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<03:12, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<03:37, 1.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<03:07, 1.88MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<02:18, 2.54MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<03:00, 1.94MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:16, 1.78MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<02:32, 2.28MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<01:50, 3.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<06:04, 949kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<04:49, 1.19MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<03:29, 1.64MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:46, 1.51MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<03:48, 1.50MB/s].vector_cache/glove.6B.zip:  60%|██████    | 522M/862M [03:46<02:54, 1.96MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:04, 2.71MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<09:58, 564kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<07:33, 744kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<05:24, 1.03MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:04, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:06, 1.35MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:59, 1.84MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:23, 1.62MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<03:26, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<02:40, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:52<01:54, 2.83MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<5:15:46, 17.2kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<3:41:20, 24.5kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<2:34:19, 34.9kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<1:48:33, 49.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<1:17:01, 69.4kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<54:01, 98.7kB/s]  .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:56<37:33, 141kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<31:38, 167kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<22:39, 233kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<15:53, 330kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<12:17, 424kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<09:06, 571kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<06:27, 802kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<05:43, 900kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<05:02, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:44, 1.37MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:02<02:39, 1.91MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<05:09, 984kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<04:08, 1.23MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<03:00, 1.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:16, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<03:18, 1.51MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:34, 1.94MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:50, 2.68MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<36:32, 135kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<26:03, 189kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<18:16, 269kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<13:49, 352kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<10:39, 457kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<07:41, 632kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<06:06, 787kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:45, 1.01MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<03:25, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:29, 1.36MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<02:50, 1.67MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<02:04, 2.27MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:13<01:31, 3.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<35:54, 130kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<26:03, 179kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<18:25, 252kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<13:29, 341kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<09:53, 464kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<06:59, 653kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<05:55, 764kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<05:03, 894kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<03:45, 1.20MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:19, 1.34MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<02:42, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:00, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:24, 1.82MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:34, 1.71MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<01:59, 2.20MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:25, 3.04MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:59, 866kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<03:56, 1.10MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:25<02:50, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:58, 1.43MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:56, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:16, 1.87MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:27<01:36, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<50:00, 83.7kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<35:22, 118kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:29<24:42, 168kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<18:07, 227kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<13:30, 305kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<09:35, 427kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<06:42, 606kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<07:04, 572kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<05:21, 754kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<03:49, 1.05MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<03:35, 1.11MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<03:19, 1.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<02:29, 1.59MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<01:46, 2.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<04:10, 939kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<03:18, 1.18MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<02:24, 1.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:34, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:34, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:59, 1.92MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<01:59, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:09, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<01:41, 2.22MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:13, 3.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<27:24, 135kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<19:32, 189kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<13:39, 269kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<10:19, 352kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<07:57, 457kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<05:42, 634kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<03:59, 895kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<3:28:33, 17.1kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<2:26:05, 24.4kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<1:41:36, 34.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<1:11:13, 49.1kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<50:32, 69.2kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<35:25, 98.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<24:28, 140kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<41:41, 82.3kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<29:28, 116kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<20:33, 165kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<15:01, 224kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<10:47, 311kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<07:34, 440kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<06:01, 546kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<04:53, 674kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:32, 925kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<02:30, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<03:14, 997kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:35, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<01:52, 1.70MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:02, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:03, 1.53MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:34, 2.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<01:07, 2.75MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<02:20, 1.32MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:57, 1.58MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<01:26, 2.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:41, 1.78MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<01:47, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:22, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<00:59, 2.99MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<02:07, 1.39MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<01:47, 1.65MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:18, 2.24MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:34, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:41, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:19, 2.17MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:21, 2.06MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:14, 2.27MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:08<00:55, 3.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:16, 2.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:26, 1.89MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:08, 2.40MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:10<00:48, 3.28MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<2:34:52, 17.3kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<1:48:25, 24.6kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<1:15:13, 35.2kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<52:33, 49.6kB/s]  .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<36:54, 70.5kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<25:38, 100kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:14<17:44, 143kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<27:35, 92.1kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<19:31, 130kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<13:34, 185kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<09:57, 248kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<07:27, 331kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<05:18, 462kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<04:01, 596kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<03:03, 783kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<02:10, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<02:02, 1.14MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<01:54, 1.23MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:26, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:21, 1.67MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:24, 1.61MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:05, 2.06MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:46, 2.83MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<16:16, 135kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<11:34, 189kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<08:03, 269kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<06:03, 352kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<04:25, 479kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<03:06, 674kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:37, 783kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<02:15, 912kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:39, 1.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:30<01:09, 1.73MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<10:36, 188kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<07:36, 261kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<05:18, 369kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<04:05, 469kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<03:14, 591kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:20, 814kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:37, 1.15MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<03:09, 587kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:23, 773kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:41, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<01:34, 1.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:27, 1.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:05, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:38<00:46, 2.25MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<01:51, 925kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:28, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:03, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<01:06, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:06, 1.48MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<00:51, 1.91MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:50, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<00:44, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:32, 2.82MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:43, 2.07MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:48, 1.85MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:37, 2.38MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:26, 3.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<01:04, 1.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:54, 1.59MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:39, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:46, 1.79MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:48, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:38, 2.14MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:38, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<00:42, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:33, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:23, 3.19MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<04:14, 292kB/s] .vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<03:04, 400kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:53<02:08, 564kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:43, 677kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:26, 808kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:03, 1.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:43, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<1:04:01, 17.2kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<44:40, 24.5kB/s]  .vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:57<30:33, 34.9kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<20:54, 49.3kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<14:48, 69.4kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<10:17, 98.7kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<06:59, 138kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<04:57, 193kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:01<03:23, 274kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<02:29, 358kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:55, 464kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<01:22, 641kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<01:02, 797kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:48, 1.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:05<00:33, 1.41MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:33, 1.37MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:27, 1.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:07<00:19, 2.20MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:22, 1.80MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:24, 1.69MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:18, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:18, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:16, 2.26MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:11<00:11, 2.98MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:15, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:13, 2.42MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:13<00:09, 3.31MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:23, 1.22MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:19, 1.48MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:13, 2.01MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<00:14, 1.71MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:15, 1.64MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<00:11, 2.09MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:10, 2.01MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:09, 2.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:06, 2.94MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:07, 2.11MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:08, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:06, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:05, 2.16MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:06, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.42MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:02, 3.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:05, 1.43MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:04, 1.70MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.30MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.85MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:02, 1.73MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:01, 2.22MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:27<00:00, 3.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 400kB/s] .vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:01<142:10:01,  1.28s/it]  0%|          | 791/400000 [00:01<99:19:29,  1.12it/s]  0%|          | 1590/400000 [00:01<69:23:32,  1.59it/s]  1%|          | 2421/400000 [00:01<48:28:38,  2.28it/s]  1%|          | 3208/400000 [00:01<33:52:16,  3.25it/s]  1%|          | 3997/400000 [00:01<23:40:00,  4.65it/s]  1%|          | 4671/400000 [00:01<16:32:36,  6.64it/s]  1%|▏         | 5464/400000 [00:01<11:33:41,  9.48it/s]  2%|▏         | 6262/400000 [00:02<8:04:50, 13.53it/s]   2%|▏         | 6996/400000 [00:02<5:39:01, 19.32it/s]  2%|▏         | 7857/400000 [00:02<3:57:01, 27.57it/s]  2%|▏         | 8733/400000 [00:02<2:45:46, 39.34it/s]  2%|▏         | 9571/400000 [00:02<1:56:01, 56.08it/s]  3%|▎         | 10471/400000 [00:02<1:21:14, 79.91it/s]  3%|▎         | 11310/400000 [00:02<57:00, 113.63it/s]   3%|▎         | 12122/400000 [00:02<40:06, 161.21it/s]  3%|▎         | 12900/400000 [00:02<28:16, 228.22it/s]  3%|▎         | 13672/400000 [00:03<20:00, 321.75it/s]  4%|▎         | 14527/400000 [00:03<14:12, 452.34it/s]  4%|▍         | 15317/400000 [00:03<10:10, 630.60it/s]  4%|▍         | 16117/400000 [00:03<07:20, 871.41it/s]  4%|▍         | 17010/400000 [00:03<05:20, 1194.87it/s]  4%|▍         | 17911/400000 [00:03<03:56, 1615.11it/s]  5%|▍         | 18811/400000 [00:03<02:57, 2142.41it/s]  5%|▍         | 19674/400000 [00:03<02:17, 2765.82it/s]  5%|▌         | 20546/400000 [00:03<01:49, 3477.63it/s]  5%|▌         | 21460/400000 [00:03<01:28, 4271.30it/s]  6%|▌         | 22340/400000 [00:04<01:14, 5037.42it/s]  6%|▌         | 23216/400000 [00:04<01:05, 5735.20it/s]  6%|▌         | 24083/400000 [00:04<01:00, 6233.03it/s]  6%|▌         | 24931/400000 [00:04<00:55, 6769.82it/s]  6%|▋         | 25775/400000 [00:04<00:53, 7011.45it/s]  7%|▋         | 26595/400000 [00:04<00:51, 7314.28it/s]  7%|▋         | 27493/400000 [00:04<00:48, 7744.35it/s]  7%|▋         | 28335/400000 [00:04<00:48, 7734.46it/s]  7%|▋         | 29168/400000 [00:04<00:46, 7903.15it/s]  7%|▋         | 29993/400000 [00:04<00:46, 7882.37it/s]  8%|▊         | 30805/400000 [00:05<00:46, 7904.93it/s]  8%|▊         | 31613/400000 [00:05<00:47, 7729.52it/s]  8%|▊         | 32411/400000 [00:05<00:47, 7798.89it/s]  8%|▊         | 33221/400000 [00:05<00:46, 7884.20it/s]  9%|▊         | 34074/400000 [00:05<00:45, 8066.44it/s]  9%|▊         | 34969/400000 [00:05<00:43, 8312.27it/s]  9%|▉         | 35806/400000 [00:05<00:45, 8041.15it/s]  9%|▉         | 36626/400000 [00:05<00:44, 8086.16it/s]  9%|▉         | 37439/400000 [00:05<00:45, 8056.21it/s] 10%|▉         | 38289/400000 [00:05<00:44, 8182.65it/s] 10%|▉         | 39189/400000 [00:06<00:42, 8409.70it/s] 10%|█         | 40034/400000 [00:06<00:42, 8413.39it/s] 10%|█         | 40878/400000 [00:06<00:43, 8252.00it/s] 10%|█         | 41706/400000 [00:06<00:45, 7866.47it/s] 11%|█         | 42499/400000 [00:06<00:46, 7647.09it/s] 11%|█         | 43269/400000 [00:06<00:47, 7535.11it/s] 11%|█         | 44027/400000 [00:06<00:47, 7506.63it/s] 11%|█         | 44824/400000 [00:06<00:46, 7637.41it/s] 11%|█▏        | 45667/400000 [00:06<00:45, 7856.16it/s] 12%|█▏        | 46456/400000 [00:07<00:45, 7846.96it/s] 12%|█▏        | 47249/400000 [00:07<00:44, 7869.81it/s] 12%|█▏        | 48038/400000 [00:07<00:45, 7726.76it/s] 12%|█▏        | 48858/400000 [00:07<00:44, 7860.49it/s] 12%|█▏        | 49646/400000 [00:07<00:44, 7864.16it/s] 13%|█▎        | 50434/400000 [00:07<00:44, 7866.10it/s] 13%|█▎        | 51255/400000 [00:07<00:43, 7963.76it/s] 13%|█▎        | 52053/400000 [00:07<00:45, 7642.89it/s] 13%|█▎        | 52821/400000 [00:07<00:45, 7612.39it/s] 13%|█▎        | 53648/400000 [00:07<00:44, 7798.34it/s] 14%|█▎        | 54432/400000 [00:08<00:44, 7810.58it/s] 14%|█▍        | 55216/400000 [00:08<00:44, 7814.89it/s] 14%|█▍        | 56020/400000 [00:08<00:43, 7880.34it/s] 14%|█▍        | 56883/400000 [00:08<00:42, 8090.33it/s] 14%|█▍        | 57783/400000 [00:08<00:41, 8342.36it/s] 15%|█▍        | 58621/400000 [00:08<00:42, 8067.54it/s] 15%|█▍        | 59433/400000 [00:08<00:42, 8046.80it/s] 15%|█▌        | 60241/400000 [00:08<00:43, 7842.38it/s] 15%|█▌        | 61029/400000 [00:08<00:43, 7804.71it/s] 15%|█▌        | 61908/400000 [00:08<00:41, 8074.11it/s] 16%|█▌        | 62720/400000 [00:09<00:41, 8082.21it/s] 16%|█▌        | 63556/400000 [00:09<00:41, 8161.05it/s] 16%|█▌        | 64433/400000 [00:09<00:40, 8332.61it/s] 16%|█▋        | 65358/400000 [00:09<00:38, 8586.17it/s] 17%|█▋        | 66221/400000 [00:09<00:39, 8472.20it/s] 17%|█▋        | 67151/400000 [00:09<00:38, 8702.84it/s] 17%|█▋        | 68028/400000 [00:09<00:38, 8720.96it/s] 17%|█▋        | 68903/400000 [00:09<00:38, 8599.35it/s] 17%|█▋        | 69766/400000 [00:09<00:38, 8608.13it/s] 18%|█▊        | 70638/400000 [00:09<00:38, 8639.32it/s] 18%|█▊        | 71505/400000 [00:10<00:37, 8646.06it/s] 18%|█▊        | 72371/400000 [00:10<00:37, 8641.77it/s] 18%|█▊        | 73236/400000 [00:10<00:38, 8466.72it/s] 19%|█▊        | 74084/400000 [00:10<00:39, 8318.76it/s] 19%|█▊        | 74920/400000 [00:10<00:39, 8330.41it/s] 19%|█▉        | 75755/400000 [00:10<00:38, 8335.35it/s] 19%|█▉        | 76645/400000 [00:10<00:38, 8496.85it/s] 19%|█▉        | 77496/400000 [00:10<00:38, 8278.19it/s] 20%|█▉        | 78326/400000 [00:10<00:40, 7973.63it/s] 20%|█▉        | 79151/400000 [00:11<00:39, 8053.67it/s] 20%|█▉        | 79960/400000 [00:11<00:39, 8024.50it/s] 20%|██        | 80765/400000 [00:11<00:41, 7691.74it/s] 20%|██        | 81539/400000 [00:11<00:42, 7467.48it/s] 21%|██        | 82291/400000 [00:11<00:44, 7120.82it/s] 21%|██        | 83013/400000 [00:11<00:44, 7150.02it/s] 21%|██        | 83806/400000 [00:11<00:42, 7364.07it/s] 21%|██        | 84549/400000 [00:11<00:42, 7382.35it/s] 21%|██▏       | 85309/400000 [00:11<00:42, 7445.92it/s] 22%|██▏       | 86111/400000 [00:11<00:41, 7607.08it/s] 22%|██▏       | 86940/400000 [00:12<00:40, 7798.39it/s] 22%|██▏       | 87723/400000 [00:12<00:40, 7750.75it/s] 22%|██▏       | 88501/400000 [00:12<00:40, 7672.35it/s] 22%|██▏       | 89324/400000 [00:12<00:39, 7831.22it/s] 23%|██▎       | 90166/400000 [00:12<00:38, 7995.74it/s] 23%|██▎       | 90986/400000 [00:12<00:38, 8054.75it/s] 23%|██▎       | 91897/400000 [00:12<00:36, 8344.53it/s] 23%|██▎       | 92736/400000 [00:12<00:37, 8173.90it/s] 23%|██▎       | 93566/400000 [00:12<00:37, 8210.40it/s] 24%|██▎       | 94439/400000 [00:12<00:36, 8356.71it/s] 24%|██▍       | 95278/400000 [00:13<00:40, 7615.11it/s] 24%|██▍       | 96073/400000 [00:13<00:39, 7710.35it/s] 24%|██▍       | 96855/400000 [00:13<00:39, 7598.28it/s] 24%|██▍       | 97757/400000 [00:13<00:37, 7975.21it/s] 25%|██▍       | 98683/400000 [00:13<00:36, 8318.73it/s] 25%|██▍       | 99544/400000 [00:13<00:35, 8403.51it/s] 25%|██▌       | 100463/400000 [00:13<00:34, 8624.10it/s] 25%|██▌       | 101333/400000 [00:13<00:35, 8395.82it/s] 26%|██▌       | 102179/400000 [00:13<00:36, 8119.82it/s] 26%|██▌       | 103016/400000 [00:14<00:36, 8191.10it/s] 26%|██▌       | 103841/400000 [00:14<00:36, 8208.27it/s] 26%|██▌       | 104666/400000 [00:14<00:36, 8111.09it/s] 26%|██▋       | 105487/400000 [00:14<00:36, 8140.25it/s] 27%|██▋       | 106389/400000 [00:14<00:35, 8383.16it/s] 27%|██▋       | 107320/400000 [00:14<00:33, 8639.39it/s] 27%|██▋       | 108189/400000 [00:14<00:33, 8639.71it/s] 27%|██▋       | 109056/400000 [00:14<00:34, 8551.03it/s] 27%|██▋       | 109914/400000 [00:14<00:34, 8516.89it/s] 28%|██▊       | 110859/400000 [00:14<00:32, 8774.71it/s] 28%|██▊       | 111767/400000 [00:15<00:32, 8862.53it/s] 28%|██▊       | 112656/400000 [00:15<00:32, 8737.11it/s] 28%|██▊       | 113541/400000 [00:15<00:32, 8769.35it/s] 29%|██▊       | 114420/400000 [00:15<00:33, 8550.80it/s] 29%|██▉       | 115278/400000 [00:15<00:33, 8544.24it/s] 29%|██▉       | 116235/400000 [00:15<00:32, 8768.67it/s] 29%|██▉       | 117157/400000 [00:15<00:31, 8896.98it/s] 30%|██▉       | 118096/400000 [00:15<00:31, 9038.54it/s] 30%|██▉       | 119002/400000 [00:15<00:32, 8563.21it/s] 30%|██▉       | 119865/400000 [00:16<00:34, 8178.90it/s] 30%|███       | 120692/400000 [00:16<00:34, 8131.12it/s] 30%|███       | 121511/400000 [00:16<00:34, 8088.98it/s] 31%|███       | 122325/400000 [00:16<00:34, 8097.69it/s] 31%|███       | 123138/400000 [00:16<00:34, 7911.31it/s] 31%|███       | 123974/400000 [00:16<00:34, 8040.40it/s] 31%|███       | 124781/400000 [00:16<00:35, 7762.59it/s] 31%|███▏      | 125583/400000 [00:16<00:35, 7835.96it/s] 32%|███▏      | 126448/400000 [00:16<00:33, 8062.06it/s] 32%|███▏      | 127327/400000 [00:16<00:32, 8266.76it/s] 32%|███▏      | 128245/400000 [00:17<00:31, 8519.90it/s] 32%|███▏      | 129102/400000 [00:17<00:32, 8396.40it/s] 32%|███▏      | 129946/400000 [00:17<00:33, 8114.51it/s] 33%|███▎      | 130763/400000 [00:17<00:33, 8073.53it/s] 33%|███▎      | 131574/400000 [00:17<00:33, 7947.62it/s] 33%|███▎      | 132372/400000 [00:17<00:33, 7922.71it/s] 33%|███▎      | 133196/400000 [00:17<00:33, 8012.72it/s] 33%|███▎      | 133999/400000 [00:17<00:34, 7711.74it/s] 34%|███▎      | 134778/400000 [00:17<00:34, 7734.20it/s] 34%|███▍      | 135644/400000 [00:17<00:33, 7988.92it/s] 34%|███▍      | 136530/400000 [00:18<00:32, 8230.06it/s] 34%|███▍      | 137508/400000 [00:18<00:30, 8639.05it/s] 35%|███▍      | 138401/400000 [00:18<00:29, 8722.97it/s] 35%|███▍      | 139280/400000 [00:18<00:31, 8341.64it/s] 35%|███▌      | 140123/400000 [00:18<00:31, 8306.89it/s] 35%|███▌      | 140990/400000 [00:18<00:30, 8409.20it/s] 35%|███▌      | 141836/400000 [00:18<00:31, 8277.09it/s] 36%|███▌      | 142668/400000 [00:18<00:31, 8150.60it/s] 36%|███▌      | 143486/400000 [00:18<00:31, 8075.32it/s] 36%|███▌      | 144296/400000 [00:19<00:31, 8034.25it/s] 36%|███▋      | 145104/400000 [00:19<00:31, 8047.82it/s] 36%|███▋      | 145910/400000 [00:19<00:31, 7987.57it/s] 37%|███▋      | 146710/400000 [00:19<00:32, 7783.30it/s] 37%|███▋      | 147536/400000 [00:19<00:31, 7918.63it/s] 37%|███▋      | 148420/400000 [00:19<00:30, 8173.92it/s] 37%|███▋      | 149241/400000 [00:19<00:31, 8037.31it/s] 38%|███▊      | 150098/400000 [00:19<00:30, 8189.66it/s] 38%|███▊      | 150920/400000 [00:19<00:30, 8143.39it/s] 38%|███▊      | 151737/400000 [00:19<00:30, 8108.42it/s] 38%|███▊      | 152571/400000 [00:20<00:30, 8175.43it/s] 38%|███▊      | 153390/400000 [00:20<00:30, 8177.87it/s] 39%|███▊      | 154209/400000 [00:20<00:30, 8152.45it/s] 39%|███▉      | 155025/400000 [00:20<00:30, 8131.89it/s] 39%|███▉      | 155839/400000 [00:20<00:31, 7852.39it/s] 39%|███▉      | 156695/400000 [00:20<00:30, 8051.16it/s] 39%|███▉      | 157629/400000 [00:20<00:28, 8397.63it/s] 40%|███▉      | 158475/400000 [00:20<00:29, 8275.68it/s] 40%|███▉      | 159308/400000 [00:20<00:29, 8238.24it/s] 40%|████      | 160136/400000 [00:20<00:29, 8040.88it/s] 40%|████      | 160944/400000 [00:21<00:31, 7678.53it/s] 40%|████      | 161718/400000 [00:21<00:31, 7656.13it/s] 41%|████      | 162488/400000 [00:21<00:32, 7356.17it/s] 41%|████      | 163275/400000 [00:21<00:31, 7499.45it/s] 41%|████      | 164030/400000 [00:21<00:31, 7463.69it/s] 41%|████      | 164872/400000 [00:21<00:30, 7726.87it/s] 41%|████▏     | 165699/400000 [00:21<00:29, 7881.84it/s] 42%|████▏     | 166531/400000 [00:21<00:29, 8006.19it/s] 42%|████▏     | 167386/400000 [00:21<00:28, 8161.43it/s] 42%|████▏     | 168206/400000 [00:22<00:28, 8150.47it/s] 42%|████▏     | 169058/400000 [00:22<00:27, 8257.33it/s] 42%|████▏     | 169895/400000 [00:22<00:27, 8288.90it/s] 43%|████▎     | 170733/400000 [00:22<00:27, 8315.02it/s] 43%|████▎     | 171578/400000 [00:22<00:27, 8352.08it/s] 43%|████▎     | 172414/400000 [00:22<00:27, 8141.40it/s] 43%|████▎     | 173230/400000 [00:22<00:28, 8034.81it/s] 44%|████▎     | 174035/400000 [00:22<00:29, 7621.14it/s] 44%|████▎     | 174803/400000 [00:22<00:29, 7613.12it/s] 44%|████▍     | 175711/400000 [00:22<00:28, 7999.80it/s] 44%|████▍     | 176534/400000 [00:23<00:27, 8065.19it/s] 44%|████▍     | 177433/400000 [00:23<00:26, 8321.88it/s] 45%|████▍     | 178356/400000 [00:23<00:25, 8573.86it/s] 45%|████▍     | 179330/400000 [00:23<00:24, 8891.05it/s] 45%|████▌     | 180265/400000 [00:23<00:24, 9022.39it/s] 45%|████▌     | 181173/400000 [00:23<00:24, 9025.35it/s] 46%|████▌     | 182162/400000 [00:23<00:23, 9265.42it/s] 46%|████▌     | 183093/400000 [00:23<00:23, 9217.49it/s] 46%|████▌     | 184033/400000 [00:23<00:23, 9270.61it/s] 46%|████▌     | 184982/400000 [00:23<00:23, 9330.70it/s] 46%|████▋     | 185917/400000 [00:24<00:23, 9284.69it/s] 47%|████▋     | 186905/400000 [00:24<00:22, 9453.42it/s] 47%|████▋     | 187852/400000 [00:24<00:22, 9239.40it/s] 47%|████▋     | 188779/400000 [00:24<00:23, 9161.40it/s] 47%|████▋     | 189697/400000 [00:24<00:22, 9154.98it/s] 48%|████▊     | 190614/400000 [00:24<00:23, 9014.14it/s] 48%|████▊     | 191578/400000 [00:24<00:22, 9192.98it/s] 48%|████▊     | 192517/400000 [00:24<00:22, 9249.20it/s] 48%|████▊     | 193444/400000 [00:24<00:23, 8893.77it/s] 49%|████▊     | 194338/400000 [00:24<00:24, 8424.14it/s] 49%|████▉     | 195189/400000 [00:25<00:24, 8207.07it/s] 49%|████▉     | 196017/400000 [00:25<00:24, 8188.50it/s] 49%|████▉     | 196895/400000 [00:25<00:24, 8355.35it/s] 49%|████▉     | 197756/400000 [00:25<00:23, 8429.14it/s] 50%|████▉     | 198602/400000 [00:25<00:23, 8434.96it/s] 50%|████▉     | 199448/400000 [00:25<00:23, 8425.15it/s] 50%|█████     | 200293/400000 [00:25<00:23, 8419.49it/s] 50%|█████     | 201136/400000 [00:25<00:24, 8282.93it/s] 50%|█████     | 201977/400000 [00:25<00:23, 8318.70it/s] 51%|█████     | 202880/400000 [00:26<00:23, 8517.41it/s] 51%|█████     | 203780/400000 [00:26<00:22, 8656.05it/s] 51%|█████     | 204700/400000 [00:26<00:22, 8811.47it/s] 51%|█████▏    | 205626/400000 [00:26<00:21, 8939.24it/s] 52%|█████▏    | 206522/400000 [00:26<00:22, 8757.66it/s] 52%|█████▏    | 207429/400000 [00:26<00:21, 8849.02it/s] 52%|█████▏    | 208316/400000 [00:26<00:21, 8781.38it/s] 52%|█████▏    | 209196/400000 [00:26<00:21, 8732.83it/s] 53%|█████▎    | 210094/400000 [00:26<00:21, 8804.81it/s] 53%|█████▎    | 210999/400000 [00:26<00:21, 8874.59it/s] 53%|█████▎    | 211959/400000 [00:27<00:20, 9079.08it/s] 53%|█████▎    | 212869/400000 [00:27<00:21, 8830.76it/s] 53%|█████▎    | 213762/400000 [00:27<00:21, 8856.73it/s] 54%|█████▎    | 214685/400000 [00:27<00:20, 8965.39it/s] 54%|█████▍    | 215584/400000 [00:27<00:20, 8885.28it/s] 54%|█████▍    | 216486/400000 [00:27<00:20, 8923.78it/s] 54%|█████▍    | 217380/400000 [00:27<00:20, 8795.20it/s] 55%|█████▍    | 218261/400000 [00:27<00:21, 8630.03it/s] 55%|█████▍    | 219126/400000 [00:27<00:21, 8368.79it/s] 55%|█████▌    | 220041/400000 [00:27<00:20, 8585.55it/s] 55%|█████▌    | 220903/400000 [00:28<00:20, 8591.69it/s] 55%|█████▌    | 221765/400000 [00:28<00:20, 8596.49it/s] 56%|█████▌    | 222704/400000 [00:28<00:20, 8819.09it/s] 56%|█████▌    | 223600/400000 [00:28<00:19, 8859.02it/s] 56%|█████▌    | 224502/400000 [00:28<00:19, 8906.56it/s] 56%|█████▋    | 225395/400000 [00:28<00:19, 8841.86it/s] 57%|█████▋    | 226281/400000 [00:28<00:19, 8690.21it/s] 57%|█████▋    | 227152/400000 [00:28<00:19, 8685.07it/s] 57%|█████▋    | 228022/400000 [00:28<00:20, 8321.80it/s] 57%|█████▋    | 228866/400000 [00:28<00:20, 8356.39it/s] 57%|█████▋    | 229777/400000 [00:29<00:19, 8567.37it/s] 58%|█████▊    | 230637/400000 [00:29<00:20, 8452.05it/s] 58%|█████▊    | 231485/400000 [00:29<00:20, 8324.01it/s] 58%|█████▊    | 232320/400000 [00:29<00:20, 8310.13it/s] 58%|█████▊    | 233178/400000 [00:29<00:19, 8374.77it/s] 59%|█████▊    | 234060/400000 [00:29<00:19, 8503.40it/s] 59%|█████▊    | 234912/400000 [00:29<00:19, 8322.11it/s] 59%|█████▉    | 235754/400000 [00:29<00:19, 8350.12it/s] 59%|█████▉    | 236631/400000 [00:29<00:19, 8471.72it/s] 59%|█████▉    | 237538/400000 [00:30<00:18, 8641.15it/s] 60%|█████▉    | 238435/400000 [00:30<00:18, 8736.28it/s] 60%|█████▉    | 239314/400000 [00:30<00:18, 8750.21it/s] 60%|██████    | 240253/400000 [00:30<00:17, 8931.16it/s] 60%|██████    | 241209/400000 [00:30<00:17, 9108.77it/s] 61%|██████    | 242144/400000 [00:30<00:17, 9177.44it/s] 61%|██████    | 243074/400000 [00:30<00:17, 9210.57it/s] 61%|██████    | 243997/400000 [00:30<00:17, 9067.94it/s] 61%|██████    | 244906/400000 [00:30<00:17, 8876.33it/s] 61%|██████▏   | 245796/400000 [00:30<00:17, 8776.50it/s] 62%|██████▏   | 246676/400000 [00:31<00:17, 8581.44it/s] 62%|██████▏   | 247537/400000 [00:31<00:18, 8447.49it/s] 62%|██████▏   | 248387/400000 [00:31<00:17, 8462.92it/s] 62%|██████▏   | 249235/400000 [00:31<00:18, 8146.51it/s] 63%|██████▎   | 250054/400000 [00:31<00:18, 8155.27it/s] 63%|██████▎   | 250872/400000 [00:31<00:18, 8101.53it/s] 63%|██████▎   | 251684/400000 [00:31<00:18, 7935.66it/s] 63%|██████▎   | 252584/400000 [00:31<00:17, 8226.62it/s] 63%|██████▎   | 253525/400000 [00:31<00:17, 8546.94it/s] 64%|██████▎   | 254445/400000 [00:31<00:16, 8732.29it/s] 64%|██████▍   | 255407/400000 [00:32<00:16, 8980.29it/s] 64%|██████▍   | 256311/400000 [00:32<00:15, 8996.65it/s] 64%|██████▍   | 257215/400000 [00:32<00:15, 8953.00it/s] 65%|██████▍   | 258130/400000 [00:32<00:15, 9008.88it/s] 65%|██████▍   | 259044/400000 [00:32<00:15, 9046.73it/s] 65%|██████▍   | 259988/400000 [00:32<00:15, 9159.07it/s] 65%|██████▌   | 260924/400000 [00:32<00:15, 9216.02it/s] 65%|██████▌   | 261847/400000 [00:32<00:16, 8543.71it/s] 66%|██████▌   | 262712/400000 [00:32<00:16, 8134.35it/s] 66%|██████▌   | 263537/400000 [00:33<00:17, 7938.94it/s] 66%|██████▌   | 264340/400000 [00:33<00:17, 7928.76it/s] 66%|██████▋   | 265140/400000 [00:33<00:17, 7884.96it/s] 66%|██████▋   | 265933/400000 [00:33<00:17, 7866.94it/s] 67%|██████▋   | 266785/400000 [00:33<00:16, 8050.35it/s] 67%|██████▋   | 267780/400000 [00:33<00:15, 8538.90it/s] 67%|██████▋   | 268740/400000 [00:33<00:14, 8831.32it/s] 67%|██████▋   | 269647/400000 [00:33<00:14, 8900.91it/s] 68%|██████▊   | 270545/400000 [00:33<00:14, 8649.93it/s] 68%|██████▊   | 271510/400000 [00:33<00:14, 8925.93it/s] 68%|██████▊   | 272482/400000 [00:34<00:13, 9149.58it/s] 68%|██████▊   | 273423/400000 [00:34<00:13, 9223.77it/s] 69%|██████▊   | 274350/400000 [00:34<00:13, 9207.90it/s] 69%|██████▉   | 275308/400000 [00:34<00:13, 9312.94it/s] 69%|██████▉   | 276279/400000 [00:34<00:13, 9427.78it/s] 69%|██████▉   | 277260/400000 [00:34<00:12, 9537.65it/s] 70%|██████▉   | 278216/400000 [00:34<00:12, 9413.41it/s] 70%|██████▉   | 279159/400000 [00:34<00:13, 8826.83it/s] 70%|███████   | 280051/400000 [00:34<00:13, 8783.08it/s] 70%|███████   | 280936/400000 [00:34<00:13, 8641.72it/s] 70%|███████   | 281805/400000 [00:35<00:14, 8427.91it/s] 71%|███████   | 282653/400000 [00:35<00:14, 8285.33it/s] 71%|███████   | 283486/400000 [00:35<00:14, 8113.09it/s] 71%|███████   | 284301/400000 [00:35<00:14, 7961.36it/s] 71%|███████▏  | 285186/400000 [00:35<00:13, 8208.31it/s] 72%|███████▏  | 286090/400000 [00:35<00:13, 8440.10it/s] 72%|███████▏  | 287005/400000 [00:35<00:13, 8638.76it/s] 72%|███████▏  | 287874/400000 [00:35<00:13, 8359.35it/s] 72%|███████▏  | 288808/400000 [00:35<00:12, 8629.56it/s] 72%|███████▏  | 289677/400000 [00:36<00:13, 8372.59it/s] 73%|███████▎  | 290520/400000 [00:36<00:13, 8255.73it/s] 73%|███████▎  | 291350/400000 [00:36<00:13, 8112.87it/s] 73%|███████▎  | 292165/400000 [00:36<00:13, 8109.93it/s] 73%|███████▎  | 292979/400000 [00:36<00:13, 7764.16it/s] 73%|███████▎  | 293774/400000 [00:36<00:13, 7816.50it/s] 74%|███████▎  | 294560/400000 [00:36<00:13, 7759.88it/s] 74%|███████▍  | 295339/400000 [00:36<00:13, 7665.57it/s] 74%|███████▍  | 296142/400000 [00:36<00:13, 7770.92it/s] 74%|███████▍  | 297012/400000 [00:36<00:12, 8027.93it/s] 74%|███████▍  | 297823/400000 [00:37<00:12, 8051.91it/s] 75%|███████▍  | 298651/400000 [00:37<00:12, 8116.70it/s] 75%|███████▍  | 299465/400000 [00:37<00:12, 7988.41it/s] 75%|███████▌  | 300289/400000 [00:37<00:12, 8059.58it/s] 75%|███████▌  | 301097/400000 [00:37<00:12, 7963.29it/s] 75%|███████▌  | 301925/400000 [00:37<00:12, 8052.97it/s] 76%|███████▌  | 302732/400000 [00:37<00:12, 8046.54it/s] 76%|███████▌  | 303538/400000 [00:37<00:12, 7672.52it/s] 76%|███████▌  | 304315/400000 [00:37<00:12, 7699.36it/s] 76%|███████▋  | 305144/400000 [00:37<00:12, 7865.46it/s] 76%|███████▋  | 305934/400000 [00:38<00:12, 7653.83it/s] 77%|███████▋  | 306800/400000 [00:38<00:11, 7928.85it/s] 77%|███████▋  | 307624/400000 [00:38<00:11, 8019.60it/s] 77%|███████▋  | 308519/400000 [00:38<00:11, 8276.79it/s] 77%|███████▋  | 309480/400000 [00:38<00:10, 8635.01it/s] 78%|███████▊  | 310351/400000 [00:38<00:11, 7944.22it/s] 78%|███████▊  | 311162/400000 [00:38<00:11, 7726.12it/s] 78%|███████▊  | 311948/400000 [00:38<00:11, 7558.84it/s] 78%|███████▊  | 312715/400000 [00:38<00:11, 7591.37it/s] 78%|███████▊  | 313562/400000 [00:39<00:11, 7834.59it/s] 79%|███████▊  | 314474/400000 [00:39<00:10, 8180.09it/s] 79%|███████▉  | 315383/400000 [00:39<00:10, 8431.34it/s] 79%|███████▉  | 316260/400000 [00:39<00:09, 8529.62it/s] 79%|███████▉  | 317155/400000 [00:39<00:09, 8649.94it/s] 80%|███████▉  | 318025/400000 [00:39<00:09, 8620.18it/s] 80%|███████▉  | 318891/400000 [00:39<00:09, 8491.48it/s] 80%|███████▉  | 319743/400000 [00:39<00:09, 8351.34it/s] 80%|████████  | 320581/400000 [00:39<00:09, 8181.90it/s] 80%|████████  | 321432/400000 [00:39<00:09, 8276.58it/s] 81%|████████  | 322294/400000 [00:40<00:09, 8374.62it/s] 81%|████████  | 323134/400000 [00:40<00:09, 8229.45it/s] 81%|████████  | 324125/400000 [00:40<00:08, 8669.99it/s] 81%|████████▏ | 325034/400000 [00:40<00:08, 8790.60it/s] 81%|████████▏ | 325964/400000 [00:40<00:08, 8936.18it/s] 82%|████████▏ | 326921/400000 [00:40<00:08, 9115.87it/s] 82%|████████▏ | 327900/400000 [00:40<00:07, 9306.42it/s] 82%|████████▏ | 328835/400000 [00:40<00:07, 9167.00it/s] 82%|████████▏ | 329755/400000 [00:40<00:07, 9050.65it/s] 83%|████████▎ | 330687/400000 [00:40<00:07, 9129.32it/s] 83%|████████▎ | 331663/400000 [00:41<00:07, 9309.31it/s] 83%|████████▎ | 332597/400000 [00:41<00:07, 8874.33it/s] 83%|████████▎ | 333533/400000 [00:41<00:07, 9012.88it/s] 84%|████████▎ | 334439/400000 [00:41<00:07, 8571.42it/s] 84%|████████▍ | 335304/400000 [00:41<00:07, 8218.86it/s] 84%|████████▍ | 336138/400000 [00:41<00:07, 8252.02it/s] 84%|████████▍ | 336970/400000 [00:41<00:07, 7974.03it/s] 84%|████████▍ | 337865/400000 [00:41<00:07, 8242.90it/s] 85%|████████▍ | 338774/400000 [00:41<00:07, 8477.86it/s] 85%|████████▍ | 339698/400000 [00:42<00:06, 8690.82it/s] 85%|████████▌ | 340651/400000 [00:42<00:06, 8926.32it/s] 85%|████████▌ | 341561/400000 [00:42<00:06, 8976.11it/s] 86%|████████▌ | 342497/400000 [00:42<00:06, 9085.53it/s] 86%|████████▌ | 343411/400000 [00:42<00:06, 9100.31it/s] 86%|████████▌ | 344349/400000 [00:42<00:06, 9182.24it/s] 86%|████████▋ | 345332/400000 [00:42<00:05, 9365.63it/s] 87%|████████▋ | 346271/400000 [00:42<00:05, 9103.41it/s] 87%|████████▋ | 347185/400000 [00:42<00:06, 8759.18it/s] 87%|████████▋ | 348066/400000 [00:42<00:05, 8730.45it/s] 87%|████████▋ | 348943/400000 [00:43<00:06, 8014.06it/s] 87%|████████▋ | 349786/400000 [00:43<00:06, 8134.28it/s] 88%|████████▊ | 350623/400000 [00:43<00:06, 8201.48it/s] 88%|████████▊ | 351451/400000 [00:43<00:06, 7800.44it/s] 88%|████████▊ | 352296/400000 [00:43<00:05, 7981.95it/s] 88%|████████▊ | 353177/400000 [00:43<00:05, 8209.14it/s] 89%|████████▊ | 354038/400000 [00:43<00:05, 8324.56it/s] 89%|████████▊ | 354876/400000 [00:43<00:05, 7860.03it/s] 89%|████████▉ | 355671/400000 [00:43<00:05, 7744.43it/s] 89%|████████▉ | 356557/400000 [00:44<00:05, 8047.55it/s] 89%|████████▉ | 357455/400000 [00:44<00:05, 8305.79it/s] 90%|████████▉ | 358360/400000 [00:44<00:04, 8514.11it/s] 90%|████████▉ | 359264/400000 [00:44<00:04, 8664.14it/s] 90%|█████████ | 360136/400000 [00:44<00:04, 8467.29it/s] 90%|█████████ | 361046/400000 [00:44<00:04, 8645.84it/s] 90%|█████████ | 361957/400000 [00:44<00:04, 8777.77it/s] 91%|█████████ | 362857/400000 [00:44<00:04, 8841.12it/s] 91%|█████████ | 363744/400000 [00:44<00:04, 8530.99it/s] 91%|█████████ | 364602/400000 [00:44<00:04, 8112.65it/s] 91%|█████████▏| 365457/400000 [00:45<00:04, 8236.99it/s] 92%|█████████▏| 366379/400000 [00:45<00:03, 8506.77it/s] 92%|█████████▏| 367312/400000 [00:45<00:03, 8735.40it/s] 92%|█████████▏| 368195/400000 [00:45<00:03, 8762.65it/s] 92%|█████████▏| 369076/400000 [00:45<00:03, 8655.69it/s] 92%|█████████▏| 369945/400000 [00:45<00:03, 8516.70it/s] 93%|█████████▎| 370800/400000 [00:45<00:03, 8427.87it/s] 93%|█████████▎| 371645/400000 [00:45<00:03, 8434.38it/s] 93%|█████████▎| 372490/400000 [00:45<00:03, 8321.63it/s] 93%|█████████▎| 373338/400000 [00:46<00:03, 8366.84it/s] 94%|█████████▎| 374251/400000 [00:46<00:03, 8579.07it/s] 94%|█████████▍| 375111/400000 [00:46<00:02, 8423.00it/s] 94%|█████████▍| 375956/400000 [00:46<00:02, 8147.28it/s] 94%|█████████▍| 376775/400000 [00:46<00:02, 7795.68it/s] 94%|█████████▍| 377602/400000 [00:46<00:02, 7930.39it/s] 95%|█████████▍| 378400/400000 [00:46<00:02, 7590.94it/s] 95%|█████████▍| 379253/400000 [00:46<00:02, 7849.27it/s] 95%|█████████▌| 380129/400000 [00:46<00:02, 8100.71it/s] 95%|█████████▌| 380983/400000 [00:46<00:02, 8222.98it/s] 95%|█████████▌| 381907/400000 [00:47<00:02, 8503.52it/s] 96%|█████████▌| 382884/400000 [00:47<00:01, 8846.16it/s] 96%|█████████▌| 383809/400000 [00:47<00:01, 8961.88it/s] 96%|█████████▌| 384714/400000 [00:47<00:01, 8986.62it/s] 96%|█████████▋| 385617/400000 [00:47<00:01, 8345.59it/s] 97%|█████████▋| 386486/400000 [00:47<00:01, 8445.29it/s] 97%|█████████▋| 387340/400000 [00:47<00:01, 8392.27it/s] 97%|█████████▋| 388186/400000 [00:47<00:01, 7998.55it/s] 97%|█████████▋| 388995/400000 [00:47<00:01, 7891.62it/s] 97%|█████████▋| 389791/400000 [00:48<00:01, 7647.25it/s] 98%|█████████▊| 390562/400000 [00:48<00:01, 7608.32it/s] 98%|█████████▊| 391331/400000 [00:48<00:01, 7631.18it/s] 98%|█████████▊| 392189/400000 [00:48<00:00, 7893.02it/s] 98%|█████████▊| 393079/400000 [00:48<00:00, 8169.14it/s] 98%|█████████▊| 393915/400000 [00:48<00:00, 8223.13it/s] 99%|█████████▊| 394796/400000 [00:48<00:00, 8388.52it/s] 99%|█████████▉| 395704/400000 [00:48<00:00, 8582.69it/s] 99%|█████████▉| 396583/400000 [00:48<00:00, 8642.98it/s] 99%|█████████▉| 397526/400000 [00:48<00:00, 8864.32it/s]100%|█████████▉| 398416/400000 [00:49<00:00, 8514.25it/s]100%|█████████▉| 399275/400000 [00:49<00:00, 8534.68it/s]100%|█████████▉| 399999/400000 [00:49<00:00, 8126.56it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1f233dabe0>, <torchtext.data.dataset.TabularDataset object at 0x7f1f233dad30>, <torchtext.vocab.Vocab object at 0x7f1f233dac50>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 13.07 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 21.54 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.78 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.78 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.78 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.96 file/s]2020-07-11 00:17:59.278583: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-11 00:17:59.283539: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-11 00:17:59.283769: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e594bdb500 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-11 00:17:59.283824: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:16, 129699.52it/s] 82%|████████▏ | 8118272/9912422 [00:00<00:09, 185157.60it/s]9920512it [00:00, 40606817.21it/s]                           
0it [00:00, ?it/s]32768it [00:00, 599586.23it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 427728.78it/s]1654784it [00:00, 10444789.62it/s]                         
0it [00:00, ?it/s]8192it [00:00, 120781.28it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

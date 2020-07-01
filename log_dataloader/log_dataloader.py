
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f56dd719ea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f56dd719ea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f57489d91e0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f57489d91e0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5766d21e18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5766d21e18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f56f5d02488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f56f5d02488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f56f5d02488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 141564.72it/s] 82%|████████▏ | 8134656/9912422 [00:00<00:08, 202083.79it/s]9920512it [00:00, 42841375.55it/s]                           
0it [00:00, ?it/s]32768it [00:00, 615633.53it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 468562.12it/s]1654784it [00:00, 11512785.84it/s]                         
0it [00:00, ?it/s]8192it [00:00, 195007.51it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f56dce2d3c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f56dce15668>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f56f5d020d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f56f5d020d0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f56f5d020d0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:26,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:25,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:25,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:24,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:24,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:23,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:22,  1.87 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:58,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:58,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:57,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:57,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:56,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:56,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:56,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:55,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:55,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:54,  2.64 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:38,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:38,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:38,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:37,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:37,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:37,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:37,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:36,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:36,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:36,  3.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:25,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:25,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:25,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:25,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:24,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:24,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:24,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:24,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:24,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:23,  5.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:16,  7.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:15,  7.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:15,  7.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:11, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:11, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:11, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:11, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:10, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:10, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:10, 10.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:06, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:06, 13.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:05, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:05, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 18.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 24.41 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 31.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 39.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 47.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 55.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 55.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 63.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 70.50 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 76.27 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 81.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 81.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 81.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 81.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.01 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.28s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.01 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 37.59 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.31s/ url]
0 examples [00:00, ? examples/s]2020-07-01 00:10:24.355119: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-01 00:10:24.369750: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-01 00:10:24.369933: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c252261fb0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-01 00:10:24.369949: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.06 examples/s]116 examples [00:00,  4.37 examples/s]229 examples [00:00,  6.23 examples/s]344 examples [00:00,  8.87 examples/s]458 examples [00:00, 12.64 examples/s]575 examples [00:00, 17.97 examples/s]687 examples [00:00, 25.49 examples/s]791 examples [00:01, 36.04 examples/s]898 examples [00:01, 50.75 examples/s]1001 examples [00:01, 70.90 examples/s]1118 examples [00:01, 98.70 examples/s]1232 examples [00:01, 135.94 examples/s]1347 examples [00:01, 184.79 examples/s]1463 examples [00:01, 247.07 examples/s]1579 examples [00:01, 323.41 examples/s]1692 examples [00:01, 407.52 examples/s]1803 examples [00:01, 497.83 examples/s]1912 examples [00:02, 589.93 examples/s]2019 examples [00:02, 679.51 examples/s]2131 examples [00:02, 770.29 examples/s]2246 examples [00:02, 854.84 examples/s]2362 examples [00:02, 927.45 examples/s]2474 examples [00:02, 973.70 examples/s]2586 examples [00:02, 974.40 examples/s]2704 examples [00:02, 1027.01 examples/s]2819 examples [00:02, 1058.55 examples/s]2931 examples [00:02, 1070.71 examples/s]3042 examples [00:03, 1078.90 examples/s]3158 examples [00:03, 1099.79 examples/s]3275 examples [00:03, 1117.95 examples/s]3389 examples [00:03, 1122.26 examples/s]3505 examples [00:03, 1131.05 examples/s]3619 examples [00:03, 1071.51 examples/s]3728 examples [00:03, 1076.89 examples/s]3843 examples [00:03, 1095.91 examples/s]3959 examples [00:03, 1114.13 examples/s]4071 examples [00:04, 1106.77 examples/s]4184 examples [00:04, 1112.97 examples/s]4296 examples [00:04, 1093.59 examples/s]4408 examples [00:04, 1100.36 examples/s]4519 examples [00:04, 1037.53 examples/s]4631 examples [00:04, 1059.98 examples/s]4744 examples [00:04, 1078.25 examples/s]4855 examples [00:04, 1085.63 examples/s]4970 examples [00:04, 1102.28 examples/s]5084 examples [00:04, 1111.36 examples/s]5197 examples [00:05, 1115.04 examples/s]5309 examples [00:05, 1116.49 examples/s]5421 examples [00:05, 1070.73 examples/s]5529 examples [00:05, 1069.25 examples/s]5637 examples [00:05, 1043.84 examples/s]5750 examples [00:05, 1054.24 examples/s]5860 examples [00:05, 1045.62 examples/s]5972 examples [00:05, 1065.16 examples/s]6085 examples [00:05, 1082.94 examples/s]6197 examples [00:05, 1093.23 examples/s]6308 examples [00:06, 1097.93 examples/s]6421 examples [00:06, 1105.67 examples/s]6536 examples [00:06, 1117.18 examples/s]6651 examples [00:06, 1124.55 examples/s]6767 examples [00:06, 1134.27 examples/s]6881 examples [00:06, 1130.15 examples/s]6995 examples [00:06, 1119.96 examples/s]7110 examples [00:06, 1126.98 examples/s]7223 examples [00:06, 1126.08 examples/s]7336 examples [00:06, 1098.78 examples/s]7447 examples [00:07, 1065.62 examples/s]7554 examples [00:07, 1059.27 examples/s]7661 examples [00:07, 1041.81 examples/s]7767 examples [00:07, 1044.72 examples/s]7883 examples [00:07, 1074.90 examples/s]7993 examples [00:07, 1080.07 examples/s]8106 examples [00:07, 1092.45 examples/s]8219 examples [00:07, 1103.01 examples/s]8330 examples [00:07, 1097.33 examples/s]8440 examples [00:08, 1087.34 examples/s]8552 examples [00:08, 1095.45 examples/s]8667 examples [00:08, 1109.91 examples/s]8780 examples [00:08, 1115.84 examples/s]8895 examples [00:08, 1122.81 examples/s]9008 examples [00:08, 1102.61 examples/s]9121 examples [00:08, 1108.48 examples/s]9232 examples [00:08, 1065.81 examples/s]9343 examples [00:08, 1077.36 examples/s]9458 examples [00:08, 1096.14 examples/s]9568 examples [00:09, 1085.73 examples/s]9677 examples [00:09, 1061.81 examples/s]9787 examples [00:09, 1071.63 examples/s]9902 examples [00:09, 1091.98 examples/s]10012 examples [00:09, 1041.49 examples/s]10117 examples [00:09, 1033.50 examples/s]10221 examples [00:09, 1015.47 examples/s]10331 examples [00:09, 1039.33 examples/s]10443 examples [00:09, 1061.83 examples/s]10553 examples [00:09, 1072.24 examples/s]10665 examples [00:10, 1085.46 examples/s]10774 examples [00:10, 1067.82 examples/s]10882 examples [00:10, 1043.50 examples/s]10991 examples [00:10, 1056.44 examples/s]11100 examples [00:10, 1064.43 examples/s]11210 examples [00:10, 1073.44 examples/s]11318 examples [00:10, 1069.69 examples/s]11429 examples [00:10, 1080.89 examples/s]11543 examples [00:10, 1096.66 examples/s]11653 examples [00:11, 1086.74 examples/s]11769 examples [00:11, 1107.37 examples/s]11881 examples [00:11, 1109.83 examples/s]11993 examples [00:11, 1108.17 examples/s]12104 examples [00:11, 1108.06 examples/s]12218 examples [00:11, 1115.60 examples/s]12331 examples [00:11, 1116.76 examples/s]12444 examples [00:11, 1119.96 examples/s]12557 examples [00:11, 1093.00 examples/s]12670 examples [00:11, 1100.90 examples/s]12783 examples [00:12, 1106.51 examples/s]12894 examples [00:12, 1069.13 examples/s]13005 examples [00:12, 1078.74 examples/s]13119 examples [00:12, 1096.09 examples/s]13236 examples [00:12, 1117.14 examples/s]13352 examples [00:12, 1128.52 examples/s]13466 examples [00:12, 1131.32 examples/s]13580 examples [00:12, 1120.68 examples/s]13695 examples [00:12, 1126.89 examples/s]13808 examples [00:12, 1115.72 examples/s]13920 examples [00:13, 1052.54 examples/s]14031 examples [00:13, 1067.32 examples/s]14139 examples [00:13, 1044.76 examples/s]14245 examples [00:13, 1025.19 examples/s]14356 examples [00:13, 1048.68 examples/s]14466 examples [00:13, 1061.39 examples/s]14579 examples [00:13, 1080.24 examples/s]14688 examples [00:13, 1049.98 examples/s]14794 examples [00:13, 1026.72 examples/s]14906 examples [00:14, 1050.73 examples/s]15022 examples [00:14, 1079.77 examples/s]15136 examples [00:14, 1093.47 examples/s]15249 examples [00:14, 1101.84 examples/s]15360 examples [00:14, 1102.80 examples/s]15472 examples [00:14, 1106.93 examples/s]15586 examples [00:14, 1115.98 examples/s]15698 examples [00:14, 1115.94 examples/s]15810 examples [00:14, 1110.20 examples/s]15922 examples [00:14, 1111.95 examples/s]16034 examples [00:15, 1102.30 examples/s]16145 examples [00:15, 1093.87 examples/s]16257 examples [00:15, 1101.40 examples/s]16370 examples [00:15, 1107.91 examples/s]16481 examples [00:15, 1100.39 examples/s]16592 examples [00:15, 1095.12 examples/s]16705 examples [00:15, 1105.20 examples/s]16820 examples [00:15, 1116.40 examples/s]16932 examples [00:15, 1116.68 examples/s]17048 examples [00:15, 1126.91 examples/s]17161 examples [00:16, 1117.34 examples/s]17277 examples [00:16, 1127.62 examples/s]17391 examples [00:16, 1130.56 examples/s]17505 examples [00:16, 1126.34 examples/s]17618 examples [00:16, 1090.86 examples/s]17732 examples [00:16, 1103.24 examples/s]17848 examples [00:16, 1116.78 examples/s]17960 examples [00:16, 1115.91 examples/s]18072 examples [00:16, 1111.95 examples/s]18184 examples [00:16, 1104.30 examples/s]18297 examples [00:17, 1109.33 examples/s]18409 examples [00:17, 1110.24 examples/s]18521 examples [00:17, 1101.84 examples/s]18632 examples [00:17, 1103.41 examples/s]18743 examples [00:17, 1085.66 examples/s]18855 examples [00:17, 1094.73 examples/s]18968 examples [00:17, 1102.62 examples/s]19082 examples [00:17, 1111.47 examples/s]19195 examples [00:17, 1116.85 examples/s]19308 examples [00:17, 1120.01 examples/s]19421 examples [00:18, 1109.55 examples/s]19538 examples [00:18, 1126.94 examples/s]19653 examples [00:18, 1133.65 examples/s]19767 examples [00:18, 1127.41 examples/s]19883 examples [00:18, 1136.64 examples/s]19997 examples [00:18, 1123.38 examples/s]20110 examples [00:18, 1065.19 examples/s]20222 examples [00:18, 1080.70 examples/s]20335 examples [00:18, 1093.53 examples/s]20445 examples [00:19, 1062.76 examples/s]20559 examples [00:19, 1082.48 examples/s]20674 examples [00:19, 1101.05 examples/s]20788 examples [00:19, 1110.04 examples/s]20902 examples [00:19, 1115.72 examples/s]21014 examples [00:19, 1049.76 examples/s]21125 examples [00:19, 1064.60 examples/s]21238 examples [00:19, 1080.59 examples/s]21347 examples [00:19, 1069.82 examples/s]21460 examples [00:19, 1084.69 examples/s]21570 examples [00:20, 1088.13 examples/s]21682 examples [00:20, 1096.02 examples/s]21793 examples [00:20, 1098.43 examples/s]21903 examples [00:20, 1097.76 examples/s]22018 examples [00:20, 1110.69 examples/s]22131 examples [00:20, 1113.78 examples/s]22243 examples [00:20, 1079.37 examples/s]22352 examples [00:20, 1074.25 examples/s]22466 examples [00:20, 1092.59 examples/s]22582 examples [00:20, 1110.35 examples/s]22694 examples [00:21, 1111.49 examples/s]22809 examples [00:21, 1122.30 examples/s]22922 examples [00:21, 1108.85 examples/s]23036 examples [00:21, 1117.45 examples/s]23151 examples [00:21, 1124.53 examples/s]23269 examples [00:21, 1139.88 examples/s]23384 examples [00:21, 1125.31 examples/s]23500 examples [00:21, 1134.67 examples/s]23615 examples [00:21, 1136.61 examples/s]23730 examples [00:21, 1139.62 examples/s]23846 examples [00:22, 1144.18 examples/s]23962 examples [00:22, 1148.35 examples/s]24084 examples [00:22, 1166.37 examples/s]24201 examples [00:22, 1143.81 examples/s]24316 examples [00:22, 1121.98 examples/s]24429 examples [00:22, 1109.71 examples/s]24541 examples [00:22, 1075.67 examples/s]24651 examples [00:22, 1081.73 examples/s]24764 examples [00:22, 1095.67 examples/s]24879 examples [00:22, 1109.84 examples/s]24992 examples [00:23, 1115.53 examples/s]25106 examples [00:23, 1122.45 examples/s]25220 examples [00:23, 1124.49 examples/s]25333 examples [00:23, 1118.51 examples/s]25445 examples [00:23, 1118.38 examples/s]25557 examples [00:23, 1097.90 examples/s]25671 examples [00:23, 1108.09 examples/s]25783 examples [00:23, 1109.72 examples/s]25895 examples [00:23, 1112.36 examples/s]26007 examples [00:24, 1112.34 examples/s]26119 examples [00:24, 1104.72 examples/s]26230 examples [00:24, 1103.45 examples/s]26341 examples [00:24, 1095.25 examples/s]26451 examples [00:24, 1073.04 examples/s]26559 examples [00:24, 1070.49 examples/s]26670 examples [00:24, 1079.74 examples/s]26783 examples [00:24, 1094.16 examples/s]26897 examples [00:24, 1107.08 examples/s]27010 examples [00:24, 1112.42 examples/s]27125 examples [00:25, 1122.73 examples/s]27240 examples [00:25, 1130.53 examples/s]27355 examples [00:25, 1136.18 examples/s]27471 examples [00:25, 1141.67 examples/s]27586 examples [00:25, 1141.09 examples/s]27701 examples [00:25, 1102.66 examples/s]27812 examples [00:25, 1100.55 examples/s]27927 examples [00:25, 1112.45 examples/s]28039 examples [00:25, 1111.41 examples/s]28151 examples [00:25, 1110.39 examples/s]28263 examples [00:26, 1091.27 examples/s]28378 examples [00:26, 1105.59 examples/s]28494 examples [00:26, 1118.40 examples/s]28611 examples [00:26, 1131.14 examples/s]28725 examples [00:26, 1124.84 examples/s]28839 examples [00:26, 1129.10 examples/s]28954 examples [00:26, 1134.74 examples/s]29068 examples [00:26, 1132.92 examples/s]29187 examples [00:26, 1148.97 examples/s]29302 examples [00:26, 1136.54 examples/s]29416 examples [00:27, 1126.92 examples/s]29533 examples [00:27, 1136.99 examples/s]29647 examples [00:27, 1124.97 examples/s]29760 examples [00:27, 1119.43 examples/s]29875 examples [00:27, 1126.83 examples/s]29991 examples [00:27, 1134.50 examples/s]30105 examples [00:27, 1046.50 examples/s]30220 examples [00:27, 1073.79 examples/s]30333 examples [00:27, 1089.67 examples/s]30445 examples [00:28, 1098.54 examples/s]30556 examples [00:28, 1085.52 examples/s]30666 examples [00:28, 1085.53 examples/s]30782 examples [00:28, 1104.20 examples/s]30896 examples [00:28, 1113.23 examples/s]31011 examples [00:28, 1121.17 examples/s]31124 examples [00:28, 1076.92 examples/s]31239 examples [00:28, 1096.06 examples/s]31350 examples [00:28, 1098.67 examples/s]31466 examples [00:28, 1114.79 examples/s]31582 examples [00:29, 1125.47 examples/s]31699 examples [00:29, 1136.24 examples/s]31813 examples [00:29, 1131.13 examples/s]31929 examples [00:29, 1139.01 examples/s]32045 examples [00:29, 1145.16 examples/s]32160 examples [00:29, 1119.91 examples/s]32273 examples [00:29, 1111.38 examples/s]32388 examples [00:29, 1122.00 examples/s]32504 examples [00:29, 1131.79 examples/s]32619 examples [00:29, 1137.02 examples/s]32733 examples [00:30, 1133.73 examples/s]32851 examples [00:30, 1144.31 examples/s]32966 examples [00:30, 1142.79 examples/s]33081 examples [00:30, 1129.81 examples/s]33196 examples [00:30, 1134.25 examples/s]33310 examples [00:30, 1133.99 examples/s]33428 examples [00:30, 1146.82 examples/s]33547 examples [00:30, 1158.51 examples/s]33665 examples [00:30, 1164.17 examples/s]33783 examples [00:30, 1167.65 examples/s]33900 examples [00:31, 1158.74 examples/s]34018 examples [00:31, 1163.46 examples/s]34135 examples [00:31, 1148.62 examples/s]34250 examples [00:31, 1130.64 examples/s]34364 examples [00:31, 1110.25 examples/s]34478 examples [00:31, 1118.95 examples/s]34591 examples [00:31, 1101.21 examples/s]34708 examples [00:31, 1118.90 examples/s]34821 examples [00:31, 1085.09 examples/s]34936 examples [00:31, 1103.76 examples/s]35047 examples [00:32, 1102.84 examples/s]35160 examples [00:32, 1110.42 examples/s]35276 examples [00:32, 1123.18 examples/s]35389 examples [00:32, 1117.95 examples/s]35504 examples [00:32, 1125.09 examples/s]35618 examples [00:32, 1128.08 examples/s]35731 examples [00:32, 1115.33 examples/s]35845 examples [00:32, 1120.77 examples/s]35958 examples [00:32, 1122.80 examples/s]36071 examples [00:33, 1112.73 examples/s]36183 examples [00:33, 1082.61 examples/s]36295 examples [00:33, 1091.68 examples/s]36409 examples [00:33, 1103.43 examples/s]36521 examples [00:33, 1105.82 examples/s]36634 examples [00:33, 1110.46 examples/s]36746 examples [00:33, 1103.39 examples/s]36861 examples [00:33, 1114.99 examples/s]36976 examples [00:33, 1124.36 examples/s]37091 examples [00:33, 1130.65 examples/s]37207 examples [00:34, 1136.46 examples/s]37321 examples [00:34, 1137.02 examples/s]37437 examples [00:34, 1141.36 examples/s]37554 examples [00:34, 1148.54 examples/s]37670 examples [00:34, 1150.15 examples/s]37786 examples [00:34, 1146.81 examples/s]37901 examples [00:34, 1134.02 examples/s]38015 examples [00:34, 1048.80 examples/s]38122 examples [00:34, 1039.65 examples/s]38235 examples [00:34, 1062.99 examples/s]38347 examples [00:35, 1078.42 examples/s]38460 examples [00:35, 1090.83 examples/s]38570 examples [00:35, 1082.23 examples/s]38681 examples [00:35, 1088.90 examples/s]38791 examples [00:35, 1066.22 examples/s]38903 examples [00:35, 1079.08 examples/s]39012 examples [00:35, 1081.17 examples/s]39126 examples [00:35, 1097.83 examples/s]39238 examples [00:35, 1102.88 examples/s]39351 examples [00:35, 1108.50 examples/s]39464 examples [00:36, 1113.94 examples/s]39576 examples [00:36, 1114.25 examples/s]39688 examples [00:36, 1079.53 examples/s]39798 examples [00:36, 1084.60 examples/s]39907 examples [00:36, 1082.98 examples/s]40016 examples [00:36, 1041.93 examples/s]40129 examples [00:36, 1064.59 examples/s]40241 examples [00:36, 1079.67 examples/s]40355 examples [00:36, 1096.62 examples/s]40468 examples [00:37, 1106.06 examples/s]40584 examples [00:37, 1119.48 examples/s]40697 examples [00:37, 1120.01 examples/s]40811 examples [00:37, 1124.91 examples/s]40924 examples [00:37, 1124.70 examples/s]41038 examples [00:37, 1128.58 examples/s]41154 examples [00:37, 1137.12 examples/s]41268 examples [00:37, 1092.30 examples/s]41378 examples [00:37, 1089.28 examples/s]41490 examples [00:37, 1096.33 examples/s]41600 examples [00:38, 1092.33 examples/s]41710 examples [00:38, 1087.31 examples/s]41819 examples [00:38, 1077.75 examples/s]41931 examples [00:38, 1088.63 examples/s]42040 examples [00:38, 1073.32 examples/s]42155 examples [00:38, 1092.78 examples/s]42266 examples [00:38, 1095.16 examples/s]42380 examples [00:38, 1107.52 examples/s]42493 examples [00:38, 1114.08 examples/s]42605 examples [00:38, 1108.85 examples/s]42716 examples [00:39, 1096.92 examples/s]42826 examples [00:39, 1066.69 examples/s]42933 examples [00:39, 1063.80 examples/s]43045 examples [00:39, 1077.97 examples/s]43154 examples [00:39, 1081.38 examples/s]43269 examples [00:39, 1100.68 examples/s]43380 examples [00:39, 1101.93 examples/s]43491 examples [00:39, 1100.32 examples/s]43602 examples [00:39, 1065.38 examples/s]43710 examples [00:39, 1068.71 examples/s]43826 examples [00:40, 1094.31 examples/s]43938 examples [00:40, 1100.84 examples/s]44052 examples [00:40, 1109.65 examples/s]44168 examples [00:40, 1123.12 examples/s]44283 examples [00:40, 1128.79 examples/s]44397 examples [00:40, 1129.48 examples/s]44511 examples [00:40, 1126.97 examples/s]44624 examples [00:40, 1114.60 examples/s]44736 examples [00:40, 1111.60 examples/s]44852 examples [00:40, 1122.84 examples/s]44968 examples [00:41, 1131.29 examples/s]45082 examples [00:41, 1126.53 examples/s]45196 examples [00:41, 1129.70 examples/s]45314 examples [00:41, 1141.64 examples/s]45429 examples [00:41, 1109.98 examples/s]45541 examples [00:41, 1109.22 examples/s]45657 examples [00:41, 1122.01 examples/s]45774 examples [00:41, 1133.72 examples/s]45891 examples [00:41, 1141.45 examples/s]46006 examples [00:42, 1123.21 examples/s]46122 examples [00:42, 1131.76 examples/s]46237 examples [00:42, 1134.63 examples/s]46354 examples [00:42, 1144.21 examples/s]46470 examples [00:42, 1146.43 examples/s]46587 examples [00:42, 1152.09 examples/s]46703 examples [00:42, 1150.89 examples/s]46819 examples [00:42, 1121.69 examples/s]46932 examples [00:42, 1111.71 examples/s]47045 examples [00:42, 1116.25 examples/s]47159 examples [00:43, 1122.25 examples/s]47272 examples [00:43, 1114.57 examples/s]47385 examples [00:43, 1117.00 examples/s]47497 examples [00:43, 1102.84 examples/s]47612 examples [00:43, 1114.64 examples/s]47724 examples [00:43, 1087.13 examples/s]47834 examples [00:43, 1090.32 examples/s]47948 examples [00:43, 1103.29 examples/s]48059 examples [00:43, 1090.41 examples/s]48169 examples [00:43, 1071.80 examples/s]48281 examples [00:44, 1084.28 examples/s]48391 examples [00:44, 1088.64 examples/s]48500 examples [00:44, 1084.69 examples/s]48609 examples [00:44, 1083.79 examples/s]48719 examples [00:44, 1086.63 examples/s]48830 examples [00:44, 1091.62 examples/s]48943 examples [00:44, 1099.25 examples/s]49057 examples [00:44, 1109.88 examples/s]49169 examples [00:44, 1097.94 examples/s]49279 examples [00:44, 1064.06 examples/s]49393 examples [00:45, 1084.70 examples/s]49506 examples [00:45, 1095.72 examples/s]49621 examples [00:45, 1108.71 examples/s]49733 examples [00:45, 1088.17 examples/s]49849 examples [00:45, 1108.55 examples/s]49962 examples [00:45, 1114.03 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 16%|█▌        | 7918/50000 [00:00<00:00, 79176.68 examples/s] 43%|████▎     | 21688/50000 [00:00<00:00, 90746.86 examples/s] 71%|███████   | 35432/50000 [00:00<00:00, 101045.15 examples/s] 99%|█████████▉| 49487/50000 [00:00<00:00, 110349.81 examples/s]                                                                0 examples [00:00, ? examples/s]97 examples [00:00, 962.74 examples/s]213 examples [00:00, 1014.10 examples/s]326 examples [00:00, 1045.10 examples/s]438 examples [00:00, 1064.64 examples/s]552 examples [00:00, 1084.26 examples/s]661 examples [00:00, 1085.00 examples/s]768 examples [00:00, 1080.20 examples/s]877 examples [00:00, 1082.34 examples/s]980 examples [00:00, 1050.07 examples/s]1097 examples [00:01, 1081.52 examples/s]1212 examples [00:01, 1101.18 examples/s]1325 examples [00:01, 1107.08 examples/s]1438 examples [00:01, 1111.61 examples/s]1557 examples [00:01, 1131.41 examples/s]1679 examples [00:01, 1154.25 examples/s]1796 examples [00:01, 1156.40 examples/s]1912 examples [00:01, 1123.49 examples/s]2027 examples [00:01, 1130.56 examples/s]2141 examples [00:01, 1132.48 examples/s]2255 examples [00:02, 1097.75 examples/s]2374 examples [00:02, 1121.80 examples/s]2487 examples [00:02, 1117.52 examples/s]2599 examples [00:02, 1092.56 examples/s]2709 examples [00:02, 1066.59 examples/s]2824 examples [00:02, 1088.97 examples/s]2937 examples [00:02, 1099.14 examples/s]3054 examples [00:02, 1117.41 examples/s]3167 examples [00:02, 1115.84 examples/s]3283 examples [00:02, 1127.76 examples/s]3403 examples [00:03, 1147.60 examples/s]3521 examples [00:03, 1154.04 examples/s]3637 examples [00:03, 1148.60 examples/s]3755 examples [00:03, 1155.80 examples/s]3871 examples [00:03, 1147.95 examples/s]3986 examples [00:03, 1147.54 examples/s]4101 examples [00:03, 1148.06 examples/s]4217 examples [00:03, 1149.28 examples/s]4332 examples [00:03, 1105.39 examples/s]4443 examples [00:03, 1062.50 examples/s]4557 examples [00:04, 1084.05 examples/s]4672 examples [00:04, 1100.60 examples/s]4787 examples [00:04, 1114.58 examples/s]4899 examples [00:04, 1099.95 examples/s]5017 examples [00:04, 1121.55 examples/s]5132 examples [00:04, 1129.27 examples/s]5249 examples [00:04, 1139.05 examples/s]5364 examples [00:04, 1135.82 examples/s]5482 examples [00:04, 1147.56 examples/s]5597 examples [00:05, 1147.98 examples/s]5714 examples [00:05, 1152.83 examples/s]5831 examples [00:05, 1156.14 examples/s]5947 examples [00:05, 1151.86 examples/s]6063 examples [00:05, 1153.79 examples/s]6179 examples [00:05, 1154.89 examples/s]6298 examples [00:05, 1164.99 examples/s]6415 examples [00:05, 1162.43 examples/s]6532 examples [00:05, 1149.95 examples/s]6648 examples [00:05, 1147.98 examples/s]6764 examples [00:06, 1150.19 examples/s]6881 examples [00:06, 1155.14 examples/s]6999 examples [00:06, 1162.04 examples/s]7116 examples [00:06, 1126.46 examples/s]7229 examples [00:06, 1119.40 examples/s]7350 examples [00:06, 1142.62 examples/s]7465 examples [00:06, 1124.26 examples/s]7586 examples [00:06, 1147.54 examples/s]7702 examples [00:06, 1149.06 examples/s]7818 examples [00:06, 1137.41 examples/s]7932 examples [00:07, 1110.42 examples/s]8048 examples [00:07, 1124.25 examples/s]8161 examples [00:07, 1108.32 examples/s]8277 examples [00:07, 1120.74 examples/s]8390 examples [00:07, 1105.65 examples/s]8505 examples [00:07, 1118.58 examples/s]8618 examples [00:07, 1110.31 examples/s]8735 examples [00:07, 1124.72 examples/s]8848 examples [00:07, 1121.64 examples/s]8961 examples [00:07, 1086.06 examples/s]9076 examples [00:08, 1101.98 examples/s]9193 examples [00:08, 1119.82 examples/s]9306 examples [00:08, 1121.65 examples/s]9419 examples [00:08, 1121.46 examples/s]9534 examples [00:08, 1127.93 examples/s]9650 examples [00:08, 1136.74 examples/s]9765 examples [00:08, 1138.99 examples/s]9881 examples [00:08, 1143.56 examples/s]9996 examples [00:08, 1134.23 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP4GBCU/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteP4GBCU/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f56f5d02488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f56f5d02488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f56f5d02488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f56902ada58>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f56901f1048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f56f5d02488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f56f5d02488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f56f5d02488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f56dce15438>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f56dce150f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f566d7b31e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f566d7b31e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f566d7b31e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f566d7b32f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f566d7b32f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f566d7b32f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.47.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.7.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.7.0)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=f4da80b88e8cc0a1038e03e1cbaf89107335bb5613b83ac82ebccb571acd55fa
  Stored in directory: /tmp/pip-ephem-wheel-cache-nwcsjqo6/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:00:52, 12.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:14:30, 16.8kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<10:01:24, 23.9kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:01:29, 34.1kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<4:54:19, 48.6kB/s].vector_cache/glove.6B.zip:   1%|          | 9.83M/862M [00:01<3:24:37, 69.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.8M/862M [00:01<2:22:29, 99.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.0M/862M [00:01<1:39:21, 141kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 24.6M/862M [00:01<1:09:11, 202kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.8M/862M [00:01<48:12, 288kB/s]  .vector_cache/glove.6B.zip:   4%|▍         | 33.4M/862M [00:01<33:43, 410kB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.0M/862M [00:02<23:31, 583kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.0M/862M [00:02<16:32, 826kB/s].vector_cache/glove.6B.zip:   6%|▌         | 47.7M/862M [00:02<11:35, 1.17MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.0M/862M [00:02<08:30, 1.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.2M/862M [00:04<07:50, 1.71MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.3M/862M [00:04<09:20, 1.44MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:04<07:18, 1.84MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.1M/862M [00:05<05:17, 2.53MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.4M/862M [00:06<08:48, 1.52MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.7M/862M [00:06<07:44, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.1M/862M [00:06<05:48, 2.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<06:49, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.7M/862M [00:08<07:43, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.4M/862M [00:08<06:08, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:09<04:27, 2.97MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:10<17:11, 769kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:10<13:24, 986kB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:10<09:41, 1.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.8M/862M [00:12<09:53, 1.33MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.0M/862M [00:12<10:00, 1.31MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.7M/862M [00:12<07:37, 1.72MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.4M/862M [00:12<05:33, 2.36MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:14<07:51, 1.66MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:14<06:59, 1.87MB/s].vector_cache/glove.6B.zip:   9%|▉         | 78.7M/862M [00:14<05:15, 2.48MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<06:26, 2.02MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<06:00, 2.16MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.9M/862M [00:16<04:34, 2.84MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<05:55, 2.18MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:18<07:09, 1.81MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.2M/862M [00:18<05:40, 2.28MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:18<04:09, 3.10MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.5M/862M [00:20<12:29, 1.03MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:20<10:14, 1.26MB/s].vector_cache/glove.6B.zip:  11%|█         | 91.2M/862M [00:20<07:31, 1.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.6M/862M [00:22<07:58, 1.61MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:22<08:33, 1.50MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<06:36, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:22<04:46, 2.67MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:24<09:34, 1.33MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<08:10, 1.56MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.5M/862M [00:24<06:04, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<06:56, 1.83MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<07:47, 1.63MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<06:03, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:26<04:24, 2.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<08:06, 1.55MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<06:54, 1.82MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<05:09, 2.43MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:28<03:50, 3.27MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:36, 994kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<11:44, 1.07MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<08:56, 1.40MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:30<06:23, 1.95MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<14:22, 867kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<11:30, 1.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<08:23, 1.48MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:29, 1.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<08:50, 1.40MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<06:55, 1.79MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:34<05:00, 2.46MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<13:42, 899kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<11:01, 1.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:36<08:03, 1.53MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:13, 1.49MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:37, 1.42MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:39, 1.84MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<04:48, 2.54MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<09:04, 1.34MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<07:43, 1.58MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:45, 2.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:34, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<07:24, 1.64MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<05:53, 2.05MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:42<04:17, 2.81MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:00, 926kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<10:29, 1.15MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<07:40, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:53, 1.52MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<08:17, 1.44MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<06:26, 1.86MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<04:39, 2.55MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<12:42, 937kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<10:16, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<07:27, 1.59MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:43, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<06:44, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<05:00, 2.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<05:59, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:03, 1.67MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:33, 2.12MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<04:02, 2.89MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<11:38, 1.00MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<09:29, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<06:58, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:19, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:26, 1.80MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:50, 2.40MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:50, 1.98MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<06:38, 1.74MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<05:13, 2.21MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:58<03:46, 3.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<09:21, 1.23MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:52, 1.46MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<05:49, 1.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:29, 1.76MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<07:11, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:41, 2.00MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<04:08, 2.75MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<12:12, 930kB/s] .vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<09:50, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<07:12, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<07:25, 1.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<07:41, 1.47MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<05:57, 1.89MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<04:19, 2.60MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<12:26, 902kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<09:57, 1.12MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<07:15, 1.54MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:25, 1.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<07:47, 1.43MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<05:59, 1.86MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<04:19, 2.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:36, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<06:36, 1.68MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:12<04:52, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<05:44, 1.92MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<06:34, 1.67MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<05:10, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<03:44, 2.93MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<10:54, 1.00MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<08:52, 1.23MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<06:30, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<06:52, 1.58MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:19, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:38, 1.92MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<04:06, 2.64MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<06:46, 1.60MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<05:56, 1.82MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<04:24, 2.44MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<05:24, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<06:15, 1.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<04:54, 2.18MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<03:34, 2.98MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<06:51, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<05:48, 1.83MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<04:24, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:24<03:14, 3.26MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<10:56, 968kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<10:00, 1.06MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<07:34, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:26<05:25, 1.94MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<20:20, 517kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<15:22, 684kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<11:01, 952kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<10:00, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<09:25, 1.11MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:07, 1.46MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<11:24, 910kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<09:09, 1.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<06:41, 1.55MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:51, 1.50MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:11, 1.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<05:33, 1.85MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:34<03:59, 2.57MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<09:56, 1.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<08:08, 1.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<05:56, 1.72MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:18, 1.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<06:46, 1.50MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<05:13, 1.94MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<03:47, 2.67MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<06:54, 1.46MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:00, 1.68MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<04:28, 2.25MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<05:15, 1.91MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<06:00, 1.67MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<04:46, 2.09MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<03:28, 2.87MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<10:50, 919kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<08:42, 1.14MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<06:19, 1.57MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:31, 1.52MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<06:44, 1.47MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<05:15, 1.88MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:47, 2.59MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<18:07, 542kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<13:46, 712kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<09:54, 989kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<08:58, 1.09MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<07:24, 1.32MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<05:24, 1.80MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<05:50, 1.66MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<05:12, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:52, 2.49MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<04:45, 2.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<05:34, 1.73MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<04:22, 2.19MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<03:09, 3.02MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<07:53, 1.21MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<06:35, 1.45MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<04:52, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:25, 1.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<04:51, 1.95MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<03:37, 2.60MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<04:31, 2.08MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<05:20, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<04:17, 2.19MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<03:07, 2.99MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<09:56, 939kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<08:02, 1.16MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:51, 1.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<06:02, 1.53MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<05:13, 1.77MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:53, 2.37MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:46, 1.93MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:19, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:13, 2.84MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<04:17, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<05:09, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:08, 2.20MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:08<03:00, 3.01MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:52, 1.54MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:09, 1.75MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:52, 2.33MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:35, 1.96MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:13, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:08, 2.17MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:12<02:58, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<11:13, 795kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<08:51, 1.01MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:26, 1.38MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<06:21, 1.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:28, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<04:03, 2.18MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:41, 1.87MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<05:20, 1.65MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<04:09, 2.11MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:02, 2.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:22, 1.62MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<04:46, 1.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:33, 2.45MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:19, 2.00MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:02, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<03:57, 2.18MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:53, 2.97MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:14, 1.64MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:36, 1.86MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<03:27, 2.47MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:17, 1.98MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:05, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:26<04:03, 2.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:26<02:56, 2.87MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<09:03, 930kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<07:15, 1.16MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<05:17, 1.59MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:31, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<05:50, 1.43MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<04:34, 1.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:30<03:18, 2.52MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<09:12, 901kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<07:24, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<05:23, 1.53MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:29, 1.50MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:45, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<04:29, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:14, 2.51MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<09:03, 900kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:18, 1.12MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<05:19, 1.52MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:25, 1.49MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:40, 1.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:22, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<03:11, 2.52MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:33, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<04:06, 1.95MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:04, 2.60MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:49, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:32, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:32, 2.24MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<02:36, 3.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<04:19, 1.82MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:52, 2.03MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:54, 2.69MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:45, 2.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:31, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:39, 2.93MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:30, 2.21MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:14, 1.82MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<03:25, 2.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:48<02:29, 3.08MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<08:06, 945kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<06:33, 1.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:49<04:47, 1.59MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:57, 1.53MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<05:04, 1.50MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:54, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:51<02:52, 2.62MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:03, 1.86MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:38, 2.06MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:44, 2.73MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:34, 2.09MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<04:13, 1.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:21, 2.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:55<02:27, 3.02MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<04:10, 1.77MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:46, 1.95MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:57<02:51, 2.58MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:32, 2.07MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<04:11, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<03:21, 2.18MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<02:25, 2.98MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<07:45, 935kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<06:12, 1.17MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:01<04:31, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:44, 1.51MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:03<04:59, 1.44MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<03:54, 1.84MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:03<02:48, 2.53MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<07:19, 971kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<05:56, 1.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<04:20, 1.63MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:32, 1.55MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<04:50, 1.46MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:47, 1.85MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:07<02:43, 2.56MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:09<07:30, 930kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:09<06:03, 1.15MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<04:24, 1.57MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:31, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<04:46, 1.44MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<03:44, 1.84MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:11<02:42, 2.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<07:34, 903kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<06:05, 1.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:13<04:25, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<04:31, 1.50MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<03:53, 1.74MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:53, 2.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<03:30, 1.91MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<03:57, 1.69MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<03:04, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:17<02:14, 2.96MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<04:01, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<03:32, 1.87MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<02:37, 2.52MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:17, 1.99MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<03:50, 1.71MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:21<03:03, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:21<02:13, 2.92MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<07:01, 924kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<05:36, 1.15MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:23<04:05, 1.58MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:16, 1.50MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<04:28, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<03:30, 1.83MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:25<02:31, 2.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<06:29, 978kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<05:13, 1.21MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:27<03:47, 1.67MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:29<04:01, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<04:16, 1.47MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:21, 1.87MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:29<02:25, 2.57MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<06:51, 906kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:31<05:30, 1.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<04:01, 1.54MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<04:06, 1.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:33<03:33, 1.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:38, 2.32MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<03:07, 1.94MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:52, 2.11MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:35<02:09, 2.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:35<01:34, 3.82MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<1:01:13, 98.0kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<44:18, 135kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<31:15, 192kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:37<21:48, 273kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<18:29, 321kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<13:37, 435kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<09:40, 611kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<07:57, 736kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<06:55, 845kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<05:08, 1.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:41<03:38, 1.59MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<05:43, 1.01MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<04:40, 1.24MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:24, 1.70MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:35, 1.60MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:45<03:43, 1.54MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:52, 1.98MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:45<02:04, 2.73MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<08:31, 663kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<06:37, 853kB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:47<04:46, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:30, 1.24MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<04:19, 1.29MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:16, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:22, 2.34MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<03:46, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:16, 1.68MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:26, 2.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:51, 1.91MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:15, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<02:35, 2.09MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:53<01:52, 2.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<05:45, 933kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<04:38, 1.16MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:23, 1.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:29, 1.52MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:40, 1.44MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:52, 1.84MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<02:04, 2.53MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<05:48, 903kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<04:38, 1.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:23, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:26, 1.50MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:40, 1.41MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<02:51, 1.80MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:01<02:03, 2.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:03<05:40, 898kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<04:33, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<03:18, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<03:21, 1.50MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<03:31, 1.42MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<02:45, 1.82MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<01:58, 2.51MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 565M/862M [04:07<05:28, 905kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<04:22, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<03:10, 1.55MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:17, 1.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<03:26, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<02:38, 1.85MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<01:55, 2.51MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:40, 1.80MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<02:26, 1.98MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:11<01:49, 2.61MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:16, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:08, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<01:37, 2.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:07, 2.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:15<02:33, 1.82MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<02:03, 2.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:15<01:29, 3.08MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<04:52, 945kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<03:55, 1.17MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:51, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:58, 1.53MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<03:07, 1.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:24, 1.88MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 593M/862M [04:19<01:43, 2.59MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:15, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<02:47, 1.60MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:04, 2.15MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:22, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<02:40, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<02:07, 2.06MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<01:31, 2.84MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<04:38, 933kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<03:45, 1.15MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:43, 1.58MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:47, 1.52MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<02:56, 1.45MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:15, 1.88MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<01:38, 2.57MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:33, 1.64MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:16, 1.85MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<01:42, 2.45MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:23, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<01:54, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<01:22, 2.95MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<04:23, 925kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:31, 1.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:33, 1.58MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:36, 1.52MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:17, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:42, 2.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:00, 1.95MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<02:18, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:48, 2.15MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:37<01:18, 2.96MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:10, 1.21MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:39, 1.44MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<01:57, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:09, 1.75MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<02:19, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<01:47, 2.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:41<01:17, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<03:16, 1.13MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<02:43, 1.36MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<01:59, 1.84MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<02:09, 1.69MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<02:21, 1.55MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<01:49, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:45<01:19, 2.71MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<03:36, 990kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:55, 1.22MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:08, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:13, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<02:21, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<01:49, 1.91MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:18, 2.64MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:28, 1.39MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:50<02:07, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:34, 2.17MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<01:47, 1.87MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<02:02, 1.65MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<01:37, 2.07MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:09, 2.83MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<03:33, 927kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:51, 1.15MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:04, 1.57MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:07, 1.52MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<02:13, 1.44MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<01:42, 1.87MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:13, 2.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:47, 1.13MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:18, 1.36MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<01:41, 1.86MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:49, 1.69MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<01:59, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<01:33, 1.96MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:07, 2.69MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<03:19, 907kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<02:39, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:55, 1.55MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<01:57, 1.51MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<01:42, 1.72MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:16, 2.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:29, 1.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:42, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:19, 2.16MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<00:57, 2.95MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<01:49, 1.54MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<01:36, 1.75MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<01:11, 2.33MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:24, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:36, 1.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:10<01:16, 2.12MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<00:55, 2.90MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:53, 922kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<02:18, 1.15MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<01:40, 1.58MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:43, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<01:45, 1.48MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:21, 1.89MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<00:58, 2.61MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<04:39, 543kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<03:31, 715kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<02:30, 998kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:16, 1.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:09, 1.14MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:38, 1.49MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<01:09, 2.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:47, 854kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:13, 1.07MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:20<01:36, 1.47MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:35, 1.45MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:40, 1.38MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:17, 1.79MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<00:55, 2.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:25, 927kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<01:57, 1.15MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<01:24, 1.58MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:26, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:30, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:26<01:09, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:26<00:50, 2.55MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:23, 1.52MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:12, 1.74MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:28<00:53, 2.33MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<01:03, 1.93MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<00:58, 2.11MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<00:43, 2.79MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<00:54, 2.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:05, 1.80MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:32<00:51, 2.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:37, 3.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:50, 1.03MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:30, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:34<01:05, 1.71MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:08, 1.61MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:13, 1.50MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<00:56, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:36<00:40, 2.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:47, 982kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:27, 1.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:03, 1.64MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:04, 1.57MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:08, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:53, 1.88MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:37, 2.58MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:48, 898kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<01:26, 1.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<01:02, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:02, 1.50MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:05, 1.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<00:49, 1.86MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<00:37, 2.47MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:44, 1.99MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<00:41, 2.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:31, 2.81MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:37, 2.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:27, 3.01MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:36, 2.24MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:43, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:35, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:50<00:24, 3.11MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:21, 937kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:05, 1.16MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:47, 1.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:47, 1.53MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:49, 1.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<00:38, 1.84MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<00:27, 2.54MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:14, 912kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:00, 1.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:42, 1.55MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:42, 1.51MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:44, 1.44MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:34, 1.85MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:23, 2.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:01, 978kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:49, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:35, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:35, 1.57MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:37, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:29, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:02<00:20, 2.58MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:57, 897kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:45, 1.12MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:32, 1.54MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:32, 1.48MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:32, 1.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:24, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:16, 2.63MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:32, 1.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:27, 1.56MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:19, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:21, 1.83MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:24, 1.62MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:18, 2.06MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:10<00:12, 2.83MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:36, 961kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:29, 1.19MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:20, 1.63MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:20, 1.54MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:21, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:16, 1.86MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:14<00:10, 2.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:26, 1.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:20, 1.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:14, 1.72MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:13, 1.61MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:14, 1.50MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 841M/862M [06:18<00:11, 1.91MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:07, 2.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:19, 920kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:15, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:10, 1.57MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.50MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:09, 1.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:07, 1.83MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:22<00:04, 2.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:11, 901kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:08, 1.13MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:05, 1.55MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:03, 1.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:03, 1.46MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:02, 1.87MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:26<00:00, 2.58MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:03, 555kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 726kB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<71:02:13,  1.56it/s]  0%|          | 885/400000 [00:00<49:37:11,  2.23it/s]  0%|          | 1649/400000 [00:00<34:40:18,  3.19it/s]  1%|          | 2406/400000 [00:00<24:13:42,  4.56it/s]  1%|          | 3269/400000 [00:01<16:55:37,  6.51it/s]  1%|          | 4069/400000 [00:01<11:49:44,  9.30it/s]  1%|          | 4897/400000 [00:01<8:16:01, 13.28it/s]   1%|▏         | 5779/400000 [00:01<5:46:39, 18.95it/s]  2%|▏         | 6584/400000 [00:01<4:02:24, 27.05it/s]  2%|▏         | 7482/400000 [00:01<2:49:31, 38.59it/s]  2%|▏         | 8404/400000 [00:01<1:58:35, 55.03it/s]  2%|▏         | 9279/400000 [00:01<1:23:03, 78.40it/s]  3%|▎         | 10178/400000 [00:01<58:13, 111.59it/s]  3%|▎         | 11088/400000 [00:01<40:52, 158.58it/s]  3%|▎         | 12006/400000 [00:02<28:45, 224.88it/s]  3%|▎         | 12898/400000 [00:02<20:18, 317.77it/s]  3%|▎         | 13787/400000 [00:02<14:23, 447.02it/s]  4%|▎         | 14685/400000 [00:02<10:16, 625.26it/s]  4%|▍         | 15574/400000 [00:02<07:23, 866.72it/s]  4%|▍         | 16469/400000 [00:02<05:22, 1188.79it/s]  4%|▍         | 17375/400000 [00:02<03:57, 1607.80it/s]  5%|▍         | 18269/400000 [00:02<02:59, 2131.53it/s]  5%|▍         | 19161/400000 [00:02<02:18, 2755.85it/s]  5%|▌         | 20056/400000 [00:02<01:49, 3477.88it/s]  5%|▌         | 20979/400000 [00:03<01:28, 4276.99it/s]  5%|▌         | 21909/400000 [00:03<01:14, 5103.65it/s]  6%|▌         | 22833/400000 [00:03<01:03, 5894.63it/s]  6%|▌         | 23746/400000 [00:03<00:57, 6579.21it/s]  6%|▌         | 24656/400000 [00:03<00:52, 7169.07it/s]  6%|▋         | 25566/400000 [00:03<00:48, 7655.97it/s]  7%|▋         | 26494/400000 [00:03<00:46, 8077.94it/s]  7%|▋         | 27409/400000 [00:03<00:44, 8361.47it/s]  7%|▋         | 28323/400000 [00:03<00:44, 8264.72it/s]  7%|▋         | 29233/400000 [00:03<00:43, 8498.58it/s]  8%|▊         | 30150/400000 [00:04<00:42, 8687.05it/s]  8%|▊         | 31075/400000 [00:04<00:41, 8846.23it/s]  8%|▊         | 31981/400000 [00:04<00:41, 8802.97it/s]  8%|▊         | 32876/400000 [00:04<00:42, 8584.67it/s]  8%|▊         | 33751/400000 [00:04<00:42, 8632.35it/s]  9%|▊         | 34663/400000 [00:04<00:41, 8771.00it/s]  9%|▉         | 35574/400000 [00:04<00:41, 8867.77it/s]  9%|▉         | 36483/400000 [00:04<00:40, 8932.61it/s]  9%|▉         | 37394/400000 [00:04<00:40, 8985.05it/s] 10%|▉         | 38296/400000 [00:04<00:40, 8993.56it/s] 10%|▉         | 39198/400000 [00:05<00:40, 8991.36it/s] 10%|█         | 40099/400000 [00:05<00:40, 8980.59it/s] 10%|█         | 40998/400000 [00:05<00:39, 8978.25it/s] 10%|█         | 41900/400000 [00:05<00:39, 8988.83it/s] 11%|█         | 42818/400000 [00:05<00:39, 9042.58it/s] 11%|█         | 43723/400000 [00:05<00:39, 9034.87it/s] 11%|█         | 44627/400000 [00:05<00:39, 9025.32it/s] 11%|█▏        | 45530/400000 [00:05<00:39, 8958.99it/s] 12%|█▏        | 46427/400000 [00:05<00:39, 8937.48it/s] 12%|█▏        | 47321/400000 [00:05<00:39, 8904.14it/s] 12%|█▏        | 48224/400000 [00:06<00:39, 8941.27it/s] 12%|█▏        | 49138/400000 [00:06<00:38, 8999.84it/s] 13%|█▎        | 50039/400000 [00:06<00:38, 9001.26it/s] 13%|█▎        | 50940/400000 [00:06<00:39, 8873.59it/s] 13%|█▎        | 51850/400000 [00:06<00:38, 8938.81it/s] 13%|█▎        | 52757/400000 [00:06<00:38, 8976.60it/s] 13%|█▎        | 53658/400000 [00:06<00:38, 8986.39it/s] 14%|█▎        | 54557/400000 [00:06<00:38, 8883.06it/s] 14%|█▍        | 55474/400000 [00:06<00:38, 8964.85it/s] 14%|█▍        | 56391/400000 [00:06<00:38, 9022.30it/s] 14%|█▍        | 57350/400000 [00:07<00:37, 9184.98it/s] 15%|█▍        | 58285/400000 [00:07<00:37, 9233.08it/s] 15%|█▍        | 59213/400000 [00:07<00:36, 9244.45it/s] 15%|█▌        | 60138/400000 [00:07<00:36, 9234.56it/s] 15%|█▌        | 61064/400000 [00:07<00:36, 9240.37it/s] 15%|█▌        | 61989/400000 [00:07<00:36, 9223.52it/s] 16%|█▌        | 62912/400000 [00:07<00:36, 9197.84it/s] 16%|█▌        | 63832/400000 [00:07<00:36, 9106.03it/s] 16%|█▌        | 64767/400000 [00:07<00:36, 9175.98it/s] 16%|█▋        | 65685/400000 [00:08<00:36, 9159.24it/s] 17%|█▋        | 66602/400000 [00:08<00:37, 9005.90it/s] 17%|█▋        | 67504/400000 [00:08<00:36, 8991.91it/s] 17%|█▋        | 68416/400000 [00:08<00:36, 9028.33it/s] 17%|█▋        | 69347/400000 [00:08<00:36, 9109.65it/s] 18%|█▊        | 70259/400000 [00:08<00:36, 9091.66it/s] 18%|█▊        | 71178/400000 [00:08<00:36, 9119.79it/s] 18%|█▊        | 72091/400000 [00:08<00:36, 9002.36it/s] 18%|█▊        | 73018/400000 [00:08<00:36, 9080.70it/s] 18%|█▊        | 73970/400000 [00:08<00:35, 9207.36it/s] 19%|█▊        | 74892/400000 [00:09<00:35, 9170.59it/s] 19%|█▉        | 75854/400000 [00:09<00:34, 9299.00it/s] 19%|█▉        | 76840/400000 [00:09<00:34, 9459.33it/s] 19%|█▉        | 77788/400000 [00:09<00:34, 9271.35it/s] 20%|█▉        | 78717/400000 [00:09<00:35, 9096.59it/s] 20%|█▉        | 79629/400000 [00:09<00:35, 9089.30it/s] 20%|██        | 80560/400000 [00:09<00:34, 9153.66it/s] 20%|██        | 81481/400000 [00:09<00:34, 9169.08it/s] 21%|██        | 82399/400000 [00:09<00:34, 9166.50it/s] 21%|██        | 83319/400000 [00:09<00:34, 9174.16it/s] 21%|██        | 84237/400000 [00:10<00:35, 9015.89it/s] 21%|██▏       | 85140/400000 [00:10<00:34, 8996.66it/s] 22%|██▏       | 86050/400000 [00:10<00:34, 9026.90it/s] 22%|██▏       | 86961/400000 [00:10<00:34, 9050.69it/s] 22%|██▏       | 87867/400000 [00:10<00:34, 8948.84it/s] 22%|██▏       | 88763/400000 [00:10<00:35, 8758.41it/s] 22%|██▏       | 89695/400000 [00:10<00:34, 8918.54it/s] 23%|██▎       | 90589/400000 [00:10<00:35, 8775.15it/s] 23%|██▎       | 91523/400000 [00:10<00:34, 8935.29it/s] 23%|██▎       | 92419/400000 [00:10<00:34, 8915.36it/s] 23%|██▎       | 93312/400000 [00:11<00:35, 8594.96it/s] 24%|██▎       | 94199/400000 [00:11<00:35, 8675.18it/s] 24%|██▍       | 95081/400000 [00:11<00:34, 8716.66it/s] 24%|██▍       | 95973/400000 [00:11<00:34, 8776.36it/s] 24%|██▍       | 96919/400000 [00:11<00:33, 8969.31it/s] 24%|██▍       | 97818/400000 [00:11<00:34, 8864.39it/s] 25%|██▍       | 98707/400000 [00:11<00:33, 8866.09it/s] 25%|██▍       | 99621/400000 [00:11<00:33, 8945.66it/s] 25%|██▌       | 100579/400000 [00:11<00:32, 9125.35it/s] 25%|██▌       | 101508/400000 [00:11<00:32, 9172.53it/s] 26%|██▌       | 102427/400000 [00:12<00:32, 9170.15it/s] 26%|██▌       | 103345/400000 [00:12<00:32, 9149.76it/s] 26%|██▌       | 104296/400000 [00:12<00:31, 9254.38it/s] 26%|██▋       | 105223/400000 [00:12<00:32, 9178.26it/s] 27%|██▋       | 106142/400000 [00:12<00:32, 9159.73it/s] 27%|██▋       | 107059/400000 [00:12<00:32, 9126.79it/s] 27%|██▋       | 107973/400000 [00:12<00:32, 9068.39it/s] 27%|██▋       | 108881/400000 [00:12<00:32, 9006.87it/s] 27%|██▋       | 109785/400000 [00:12<00:32, 9014.97it/s] 28%|██▊       | 110732/400000 [00:12<00:31, 9144.54it/s] 28%|██▊       | 111650/400000 [00:13<00:31, 9154.53it/s] 28%|██▊       | 112600/400000 [00:13<00:31, 9254.31it/s] 28%|██▊       | 113526/400000 [00:13<00:31, 9236.87it/s] 29%|██▊       | 114459/400000 [00:13<00:30, 9264.19it/s] 29%|██▉       | 115406/400000 [00:13<00:30, 9316.81it/s] 29%|██▉       | 116338/400000 [00:13<00:30, 9252.81it/s] 29%|██▉       | 117317/400000 [00:13<00:30, 9407.44it/s] 30%|██▉       | 118271/400000 [00:13<00:29, 9446.77it/s] 30%|██▉       | 119217/400000 [00:13<00:29, 9399.14it/s] 30%|███       | 120158/400000 [00:13<00:30, 9319.94it/s] 30%|███       | 121091/400000 [00:14<00:30, 9213.17it/s] 31%|███       | 122017/400000 [00:14<00:30, 9226.94it/s] 31%|███       | 122941/400000 [00:14<00:30, 9224.00it/s] 31%|███       | 123864/400000 [00:14<00:30, 8909.69it/s] 31%|███       | 124758/400000 [00:14<00:31, 8821.65it/s] 31%|███▏      | 125683/400000 [00:14<00:30, 8943.60it/s] 32%|███▏      | 126642/400000 [00:14<00:29, 9126.51it/s] 32%|███▏      | 127557/400000 [00:14<00:29, 9115.92it/s] 32%|███▏      | 128473/400000 [00:14<00:29, 9128.19it/s] 32%|███▏      | 129387/400000 [00:15<00:29, 9032.62it/s] 33%|███▎      | 130305/400000 [00:15<00:29, 9073.88it/s] 33%|███▎      | 131242/400000 [00:15<00:29, 9160.65it/s] 33%|███▎      | 132187/400000 [00:15<00:28, 9245.52it/s] 33%|███▎      | 133113/400000 [00:15<00:29, 9067.26it/s] 34%|███▎      | 134040/400000 [00:15<00:29, 9125.66it/s] 34%|███▎      | 134959/400000 [00:15<00:28, 9141.99it/s] 34%|███▍      | 135882/400000 [00:15<00:28, 9167.22it/s] 34%|███▍      | 136800/400000 [00:15<00:29, 9016.86it/s] 34%|███▍      | 137760/400000 [00:15<00:28, 9181.62it/s] 35%|███▍      | 138712/400000 [00:16<00:28, 9280.35it/s] 35%|███▍      | 139642/400000 [00:16<00:28, 9089.13it/s] 35%|███▌      | 140553/400000 [00:16<00:28, 8950.98it/s] 35%|███▌      | 141450/400000 [00:16<00:29, 8845.34it/s] 36%|███▌      | 142336/400000 [00:16<00:29, 8810.30it/s] 36%|███▌      | 143245/400000 [00:16<00:28, 8892.09it/s] 36%|███▌      | 144148/400000 [00:16<00:28, 8932.56it/s] 36%|███▋      | 145042/400000 [00:16<00:28, 8929.44it/s] 36%|███▋      | 145974/400000 [00:16<00:28, 9043.04it/s] 37%|███▋      | 146923/400000 [00:16<00:27, 9171.88it/s] 37%|███▋      | 147843/400000 [00:17<00:27, 9177.85it/s] 37%|███▋      | 148765/400000 [00:17<00:27, 9188.69it/s] 37%|███▋      | 149685/400000 [00:17<00:27, 9039.60it/s] 38%|███▊      | 150590/400000 [00:17<00:27, 8912.89it/s] 38%|███▊      | 151483/400000 [00:17<00:28, 8845.99it/s] 38%|███▊      | 152423/400000 [00:17<00:27, 9003.35it/s] 38%|███▊      | 153325/400000 [00:17<00:28, 8785.91it/s] 39%|███▊      | 154215/400000 [00:17<00:27, 8817.78it/s] 39%|███▉      | 155099/400000 [00:17<00:27, 8791.34it/s] 39%|███▉      | 155985/400000 [00:17<00:27, 8809.69it/s] 39%|███▉      | 156874/400000 [00:18<00:27, 8831.25it/s] 39%|███▉      | 157801/400000 [00:18<00:27, 8957.17it/s] 40%|███▉      | 158755/400000 [00:18<00:26, 9124.31it/s] 40%|███▉      | 159691/400000 [00:18<00:26, 9191.64it/s] 40%|████      | 160629/400000 [00:18<00:25, 9246.07it/s] 40%|████      | 161555/400000 [00:18<00:25, 9219.12it/s] 41%|████      | 162478/400000 [00:18<00:26, 8978.31it/s] 41%|████      | 163382/400000 [00:18<00:26, 8996.71it/s] 41%|████      | 164293/400000 [00:18<00:26, 9029.51it/s] 41%|████▏     | 165244/400000 [00:18<00:25, 9168.30it/s] 42%|████▏     | 166162/400000 [00:19<00:25, 9144.32it/s] 42%|████▏     | 167078/400000 [00:19<00:25, 9102.44it/s] 42%|████▏     | 167998/400000 [00:19<00:25, 9128.84it/s] 42%|████▏     | 168932/400000 [00:19<00:25, 9191.03it/s] 42%|████▏     | 169867/400000 [00:19<00:24, 9236.46it/s] 43%|████▎     | 170791/400000 [00:19<00:25, 9090.89it/s] 43%|████▎     | 171718/400000 [00:19<00:24, 9142.98it/s] 43%|████▎     | 172645/400000 [00:19<00:24, 9180.64it/s] 43%|████▎     | 173599/400000 [00:19<00:24, 9284.61it/s] 44%|████▎     | 174535/400000 [00:19<00:24, 9306.62it/s] 44%|████▍     | 175467/400000 [00:20<00:24, 9308.04it/s] 44%|████▍     | 176399/400000 [00:20<00:24, 9282.26it/s] 44%|████▍     | 177343/400000 [00:20<00:23, 9326.62it/s] 45%|████▍     | 178276/400000 [00:20<00:23, 9275.63it/s] 45%|████▍     | 179218/400000 [00:20<00:23, 9317.35it/s] 45%|████▌     | 180157/400000 [00:20<00:23, 9337.08it/s] 45%|████▌     | 181096/400000 [00:20<00:23, 9352.42it/s] 46%|████▌     | 182032/400000 [00:20<00:23, 9314.14it/s] 46%|████▌     | 182964/400000 [00:20<00:23, 9279.45it/s] 46%|████▌     | 183904/400000 [00:20<00:23, 9314.54it/s] 46%|████▌     | 184836/400000 [00:21<00:23, 9233.40it/s] 46%|████▋     | 185760/400000 [00:21<00:23, 9207.25it/s] 47%|████▋     | 186681/400000 [00:21<00:23, 9150.18it/s] 47%|████▋     | 187597/400000 [00:21<00:23, 9069.11it/s] 47%|████▋     | 188521/400000 [00:21<00:23, 9119.24it/s] 47%|████▋     | 189448/400000 [00:21<00:22, 9163.62it/s] 48%|████▊     | 190382/400000 [00:21<00:22, 9215.27it/s] 48%|████▊     | 191304/400000 [00:21<00:22, 9117.04it/s] 48%|████▊     | 192217/400000 [00:21<00:22, 9081.10it/s] 48%|████▊     | 193156/400000 [00:22<00:22, 9171.39it/s] 49%|████▊     | 194106/400000 [00:22<00:22, 9266.52it/s] 49%|████▉     | 195043/400000 [00:22<00:22, 9297.12it/s] 49%|████▉     | 195985/400000 [00:22<00:21, 9333.08it/s] 49%|████▉     | 196920/400000 [00:22<00:21, 9334.58it/s] 49%|████▉     | 197889/400000 [00:22<00:21, 9436.55it/s] 50%|████▉     | 198864/400000 [00:22<00:21, 9527.25it/s] 50%|████▉     | 199818/400000 [00:22<00:21, 9452.69it/s] 50%|█████     | 200764/400000 [00:22<00:21, 9345.48it/s] 50%|█████     | 201700/400000 [00:22<00:21, 9345.65it/s] 51%|█████     | 202658/400000 [00:23<00:20, 9414.52it/s] 51%|█████     | 203600/400000 [00:23<00:21, 9270.87it/s] 51%|█████     | 204528/400000 [00:23<00:21, 9191.57it/s] 51%|█████▏    | 205459/400000 [00:23<00:21, 9225.97it/s] 52%|█████▏    | 206410/400000 [00:23<00:20, 9307.83it/s] 52%|█████▏    | 207342/400000 [00:23<00:20, 9307.55it/s] 52%|█████▏    | 208291/400000 [00:23<00:20, 9359.23it/s] 52%|█████▏    | 209228/400000 [00:23<00:20, 9310.22it/s] 53%|█████▎    | 210170/400000 [00:23<00:20, 9342.87it/s] 53%|█████▎    | 211120/400000 [00:23<00:20, 9389.04it/s] 53%|█████▎    | 212060/400000 [00:24<00:20, 9352.15it/s] 53%|█████▎    | 212996/400000 [00:24<00:20, 9326.81it/s] 53%|█████▎    | 213929/400000 [00:24<00:20, 9272.03it/s] 54%|█████▎    | 214862/400000 [00:24<00:19, 9287.04it/s] 54%|█████▍    | 215798/400000 [00:24<00:19, 9308.59it/s] 54%|█████▍    | 216753/400000 [00:24<00:19, 9376.36it/s] 54%|█████▍    | 217691/400000 [00:24<00:19, 9293.69it/s] 55%|█████▍    | 218628/400000 [00:24<00:19, 9313.73it/s] 55%|█████▍    | 219560/400000 [00:24<00:19, 9269.23it/s] 55%|█████▌    | 220488/400000 [00:24<00:20, 8932.06it/s] 55%|█████▌    | 221384/400000 [00:25<00:20, 8918.22it/s] 56%|█████▌    | 222304/400000 [00:25<00:19, 8999.77it/s] 56%|█████▌    | 223237/400000 [00:25<00:19, 9094.51it/s] 56%|█████▌    | 224165/400000 [00:25<00:19, 9148.90it/s] 56%|█████▋    | 225116/400000 [00:25<00:18, 9251.94it/s] 57%|█████▋    | 226043/400000 [00:25<00:18, 9227.03it/s] 57%|█████▋    | 226967/400000 [00:25<00:18, 9229.34it/s] 57%|█████▋    | 227891/400000 [00:25<00:18, 9215.67it/s] 57%|█████▋    | 228838/400000 [00:25<00:18, 9288.20it/s] 57%|█████▋    | 229768/400000 [00:25<00:18, 9203.57it/s] 58%|█████▊    | 230692/400000 [00:26<00:18, 9214.39it/s] 58%|█████▊    | 231663/400000 [00:26<00:17, 9356.01it/s] 58%|█████▊    | 232600/400000 [00:26<00:17, 9314.19it/s] 58%|█████▊    | 233540/400000 [00:26<00:17, 9337.33it/s] 59%|█████▊    | 234475/400000 [00:26<00:17, 9253.89it/s] 59%|█████▉    | 235407/400000 [00:26<00:17, 9273.07it/s] 59%|█████▉    | 236335/400000 [00:26<00:17, 9228.61it/s] 59%|█████▉    | 237259/400000 [00:26<00:17, 9184.28it/s] 60%|█████▉    | 238178/400000 [00:26<00:17, 9165.96it/s] 60%|█████▉    | 239095/400000 [00:26<00:17, 9063.06it/s] 60%|██████    | 240070/400000 [00:27<00:17, 9257.59it/s] 60%|██████    | 241006/400000 [00:27<00:17, 9285.31it/s] 60%|██████    | 241936/400000 [00:27<00:17, 9238.18it/s] 61%|██████    | 242900/400000 [00:27<00:16, 9354.16it/s] 61%|██████    | 243837/400000 [00:27<00:16, 9265.58it/s] 61%|██████    | 244794/400000 [00:27<00:16, 9353.37it/s] 61%|██████▏   | 245739/400000 [00:27<00:16, 9380.11it/s] 62%|██████▏   | 246678/400000 [00:27<00:16, 9370.59it/s] 62%|██████▏   | 247616/400000 [00:27<00:16, 9305.68it/s] 62%|██████▏   | 248550/400000 [00:27<00:16, 9313.07it/s] 62%|██████▏   | 249486/400000 [00:28<00:16, 9326.11it/s] 63%|██████▎   | 250420/400000 [00:28<00:16, 9328.16it/s] 63%|██████▎   | 251353/400000 [00:28<00:16, 9122.04it/s] 63%|██████▎   | 252267/400000 [00:28<00:16, 8786.47it/s] 63%|██████▎   | 253199/400000 [00:28<00:16, 8939.40it/s] 64%|██████▎   | 254152/400000 [00:28<00:16, 9108.26it/s] 64%|██████▍   | 255130/400000 [00:28<00:15, 9297.85it/s] 64%|██████▍   | 256063/400000 [00:28<00:16, 8981.97it/s] 64%|██████▍   | 256981/400000 [00:28<00:15, 9038.19it/s] 64%|██████▍   | 257928/400000 [00:29<00:15, 9161.11it/s] 65%|██████▍   | 258847/400000 [00:29<00:15, 9082.84it/s] 65%|██████▍   | 259801/400000 [00:29<00:15, 9212.37it/s] 65%|██████▌   | 260725/400000 [00:29<00:15, 9168.32it/s] 65%|██████▌   | 261709/400000 [00:29<00:14, 9359.02it/s] 66%|██████▌   | 262657/400000 [00:29<00:14, 9394.95it/s] 66%|██████▌   | 263604/400000 [00:29<00:14, 9414.67it/s] 66%|██████▌   | 264547/400000 [00:29<00:14, 9360.33it/s] 66%|██████▋   | 265484/400000 [00:29<00:14, 9296.72it/s] 67%|██████▋   | 266415/400000 [00:29<00:14, 9235.69it/s] 67%|██████▋   | 267340/400000 [00:30<00:14, 8900.83it/s] 67%|██████▋   | 268250/400000 [00:30<00:14, 8958.60it/s] 67%|██████▋   | 269149/400000 [00:30<00:14, 8883.19it/s] 68%|██████▊   | 270082/400000 [00:30<00:14, 9011.15it/s] 68%|██████▊   | 270985/400000 [00:30<00:14, 8866.88it/s] 68%|██████▊   | 271929/400000 [00:30<00:14, 9029.73it/s] 68%|██████▊   | 272863/400000 [00:30<00:13, 9118.86it/s] 68%|██████▊   | 273778/400000 [00:30<00:13, 9127.88it/s] 69%|██████▊   | 274692/400000 [00:30<00:14, 8860.38it/s] 69%|██████▉   | 275615/400000 [00:30<00:13, 8966.31it/s] 69%|██████▉   | 276514/400000 [00:31<00:13, 8962.72it/s] 69%|██████▉   | 277412/400000 [00:31<00:13, 8851.37it/s] 70%|██████▉   | 278337/400000 [00:31<00:13, 8966.67it/s] 70%|██████▉   | 279235/400000 [00:31<00:13, 8904.30it/s] 70%|███████   | 280133/400000 [00:31<00:13, 8924.46it/s] 70%|███████   | 281060/400000 [00:31<00:13, 9025.28it/s] 70%|███████   | 281964/400000 [00:31<00:13, 8827.79it/s] 71%|███████   | 282849/400000 [00:31<00:13, 8823.18it/s] 71%|███████   | 283733/400000 [00:31<00:13, 8697.03it/s] 71%|███████   | 284634/400000 [00:31<00:13, 8788.10it/s] 71%|███████▏  | 285557/400000 [00:32<00:12, 8915.17it/s] 72%|███████▏  | 286464/400000 [00:32<00:12, 8961.03it/s] 72%|███████▏  | 287406/400000 [00:32<00:12, 9091.62it/s] 72%|███████▏  | 288319/400000 [00:32<00:12, 9101.01it/s] 72%|███████▏  | 289230/400000 [00:32<00:12, 9069.35it/s] 73%|███████▎  | 290138/400000 [00:32<00:12, 8938.93it/s] 73%|███████▎  | 291056/400000 [00:32<00:12, 9009.07it/s] 73%|███████▎  | 291979/400000 [00:32<00:11, 9072.83it/s] 73%|███████▎  | 292900/400000 [00:32<00:11, 9111.50it/s] 73%|███████▎  | 293837/400000 [00:32<00:11, 9186.05it/s] 74%|███████▎  | 294757/400000 [00:33<00:11, 9125.42it/s] 74%|███████▍  | 295670/400000 [00:33<00:11, 9094.74it/s] 74%|███████▍  | 296588/400000 [00:33<00:11, 9118.87it/s] 74%|███████▍  | 297501/400000 [00:33<00:11, 8905.28it/s] 75%|███████▍  | 298427/400000 [00:33<00:11, 9007.60it/s] 75%|███████▍  | 299346/400000 [00:33<00:11, 9058.04it/s] 75%|███████▌  | 300259/400000 [00:33<00:10, 9077.03it/s] 75%|███████▌  | 301201/400000 [00:33<00:10, 9176.74it/s] 76%|███████▌  | 302120/400000 [00:33<00:10, 9170.75it/s] 76%|███████▌  | 303038/400000 [00:33<00:10, 9127.48it/s] 76%|███████▌  | 303956/400000 [00:34<00:10, 9142.33it/s] 76%|███████▌  | 304891/400000 [00:34<00:10, 9202.92it/s] 76%|███████▋  | 305862/400000 [00:34<00:10, 9347.37it/s] 77%|███████▋  | 306798/400000 [00:34<00:10, 9290.31it/s] 77%|███████▋  | 307728/400000 [00:34<00:09, 9281.08it/s] 77%|███████▋  | 308686/400000 [00:34<00:09, 9367.21it/s] 77%|███████▋  | 309624/400000 [00:34<00:09, 9165.76it/s] 78%|███████▊  | 310556/400000 [00:34<00:09, 9210.44it/s] 78%|███████▊  | 311479/400000 [00:34<00:09, 9138.68it/s] 78%|███████▊  | 312394/400000 [00:35<00:09, 8988.13it/s] 78%|███████▊  | 313294/400000 [00:35<00:09, 8767.24it/s] 79%|███████▊  | 314208/400000 [00:35<00:09, 8875.13it/s] 79%|███████▉  | 315146/400000 [00:35<00:09, 9020.58it/s] 79%|███████▉  | 316050/400000 [00:35<00:09, 9007.58it/s] 79%|███████▉  | 317004/400000 [00:35<00:09, 9160.28it/s] 79%|███████▉  | 317972/400000 [00:35<00:08, 9308.74it/s] 80%|███████▉  | 318907/400000 [00:35<00:08, 9320.29it/s] 80%|███████▉  | 319843/400000 [00:35<00:08, 9329.66it/s] 80%|████████  | 320777/400000 [00:35<00:08, 9246.00it/s] 80%|████████  | 321711/400000 [00:36<00:08, 9273.45it/s] 81%|████████  | 322639/400000 [00:36<00:08, 9263.25it/s] 81%|████████  | 323577/400000 [00:36<00:08, 9297.63it/s] 81%|████████  | 324508/400000 [00:36<00:08, 9284.08it/s] 81%|████████▏ | 325437/400000 [00:36<00:08, 9262.85it/s] 82%|████████▏ | 326364/400000 [00:36<00:07, 9251.74it/s] 82%|████████▏ | 327290/400000 [00:36<00:07, 9234.44it/s] 82%|████████▏ | 328222/400000 [00:36<00:07, 9259.91it/s] 82%|████████▏ | 329149/400000 [00:36<00:07, 9233.26it/s] 83%|████████▎ | 330073/400000 [00:36<00:07, 9138.40it/s] 83%|████████▎ | 330988/400000 [00:37<00:07, 9020.03it/s] 83%|████████▎ | 331891/400000 [00:37<00:07, 8876.17it/s] 83%|████████▎ | 332791/400000 [00:37<00:07, 8910.85it/s] 83%|████████▎ | 333721/400000 [00:37<00:07, 9023.04it/s] 84%|████████▎ | 334639/400000 [00:37<00:07, 9069.45it/s] 84%|████████▍ | 335572/400000 [00:37<00:07, 9144.91it/s] 84%|████████▍ | 336488/400000 [00:37<00:06, 9141.80it/s] 84%|████████▍ | 337403/400000 [00:37<00:06, 9118.93it/s] 85%|████████▍ | 338330/400000 [00:37<00:06, 9162.06it/s] 85%|████████▍ | 339247/400000 [00:37<00:06, 9152.71it/s] 85%|████████▌ | 340183/400000 [00:38<00:06, 9212.54it/s] 85%|████████▌ | 341132/400000 [00:38<00:06, 9291.59it/s] 86%|████████▌ | 342070/400000 [00:38<00:06, 9315.83it/s] 86%|████████▌ | 343032/400000 [00:38<00:06, 9403.39it/s] 86%|████████▌ | 343973/400000 [00:38<00:06, 9212.62it/s] 86%|████████▌ | 344896/400000 [00:38<00:06, 9068.94it/s] 86%|████████▋ | 345838/400000 [00:38<00:05, 9170.84it/s] 87%|████████▋ | 346773/400000 [00:38<00:05, 9221.36it/s] 87%|████████▋ | 347737/400000 [00:38<00:05, 9340.87it/s] 87%|████████▋ | 348673/400000 [00:38<00:05, 9339.82it/s] 87%|████████▋ | 349608/400000 [00:39<00:05, 9337.70it/s] 88%|████████▊ | 350543/400000 [00:39<00:05, 9306.30it/s] 88%|████████▊ | 351517/400000 [00:39<00:05, 9431.30it/s] 88%|████████▊ | 352461/400000 [00:39<00:05, 9377.51it/s] 88%|████████▊ | 353400/400000 [00:39<00:04, 9346.52it/s] 89%|████████▊ | 354336/400000 [00:39<00:04, 9202.81it/s] 89%|████████▉ | 355258/400000 [00:39<00:04, 9195.29it/s] 89%|████████▉ | 356207/400000 [00:39<00:04, 9281.16it/s] 89%|████████▉ | 357136/400000 [00:39<00:04, 9210.27it/s] 90%|████████▉ | 358058/400000 [00:39<00:04, 9122.40it/s] 90%|████████▉ | 358971/400000 [00:40<00:04, 9098.65it/s] 90%|████████▉ | 359882/400000 [00:40<00:04, 9064.16it/s] 90%|█████████ | 360796/400000 [00:40<00:04, 9085.39it/s] 90%|█████████ | 361705/400000 [00:40<00:04, 9080.46it/s] 91%|█████████ | 362614/400000 [00:40<00:04, 8931.27it/s] 91%|█████████ | 363535/400000 [00:40<00:04, 9011.24it/s] 91%|█████████ | 364468/400000 [00:40<00:03, 9104.23it/s] 91%|█████████▏| 365401/400000 [00:40<00:03, 9168.06it/s] 92%|█████████▏| 366328/400000 [00:40<00:03, 9195.55it/s] 92%|█████████▏| 367248/400000 [00:40<00:03, 9138.11it/s] 92%|█████████▏| 368163/400000 [00:41<00:03, 9064.46it/s] 92%|█████████▏| 369090/400000 [00:41<00:03, 9124.84it/s] 93%|█████████▎| 370029/400000 [00:41<00:03, 9201.46it/s] 93%|█████████▎| 370950/400000 [00:41<00:03, 9187.41it/s] 93%|█████████▎| 371870/400000 [00:41<00:03, 9062.27it/s] 93%|█████████▎| 372777/400000 [00:41<00:03, 9039.35it/s] 93%|█████████▎| 373682/400000 [00:41<00:02, 9020.45it/s] 94%|█████████▎| 374589/400000 [00:41<00:02, 9032.57it/s] 94%|█████████▍| 375493/400000 [00:41<00:02, 8937.08it/s] 94%|█████████▍| 376393/400000 [00:41<00:02, 8954.53it/s] 94%|█████████▍| 377302/400000 [00:42<00:02, 8994.65it/s] 95%|█████████▍| 378223/400000 [00:42<00:02, 9056.42it/s] 95%|█████████▍| 379134/400000 [00:42<00:02, 9071.92it/s] 95%|█████████▌| 380042/400000 [00:42<00:02, 8882.86it/s] 95%|█████████▌| 380964/400000 [00:42<00:02, 8979.27it/s] 95%|█████████▌| 381863/400000 [00:42<00:02, 8874.74it/s] 96%|█████████▌| 382752/400000 [00:42<00:01, 8837.35it/s] 96%|█████████▌| 383662/400000 [00:42<00:01, 8913.19it/s] 96%|█████████▌| 384554/400000 [00:42<00:01, 8862.87it/s] 96%|█████████▋| 385532/400000 [00:43<00:01, 9117.47it/s] 97%|█████████▋| 386446/400000 [00:43<00:01, 9096.64it/s] 97%|█████████▋| 387358/400000 [00:43<00:01, 9080.08it/s] 97%|█████████▋| 388290/400000 [00:43<00:01, 9149.90it/s] 97%|█████████▋| 389229/400000 [00:43<00:01, 9219.16it/s] 98%|█████████▊| 390159/400000 [00:43<00:01, 9242.11it/s] 98%|█████████▊| 391093/400000 [00:43<00:00, 9270.31it/s] 98%|█████████▊| 392023/400000 [00:43<00:00, 9277.72it/s] 98%|█████████▊| 392961/400000 [00:43<00:00, 9305.38it/s] 98%|█████████▊| 393912/400000 [00:43<00:00, 9365.01it/s] 99%|█████████▊| 394849/400000 [00:44<00:00, 9322.61it/s] 99%|█████████▉| 395782/400000 [00:44<00:00, 9185.78it/s] 99%|█████████▉| 396702/400000 [00:44<00:00, 9183.75it/s] 99%|█████████▉| 397689/400000 [00:44<00:00, 9377.75it/s]100%|█████████▉| 398662/400000 [00:44<00:00, 9479.60it/s]100%|█████████▉| 399612/400000 [00:44<00:00, 9443.28it/s]100%|█████████▉| 399999/400000 [00:44<00:00, 8975.20it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f566d7c6438>, <torchtext.data.dataset.TabularDataset object at 0x7f566d7c6588>, <torchtext.vocab.Vocab object at 0x7f566d7c64a8>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 29.22 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 41.29 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 26.09 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 26.09 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.21 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  8.21 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  9.22 file/s]2020-07-01 00:19:18.611613: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-07-01 00:19:18.616325: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-07-01 00:19:18.617062: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x556740946f20 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-01 00:19:18.617078: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 139598.26it/s] 83%|████████▎ | 8192000/9912422 [00:00<00:08, 199277.84it/s]9920512it [00:00, 42481577.54it/s]                           
0it [00:00, ?it/s]32768it [00:00, 517010.44it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 445652.56it/s]1654784it [00:00, 11545514.98it/s]                         
0it [00:00, ?it/s]8192it [00:00, 144967.40it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

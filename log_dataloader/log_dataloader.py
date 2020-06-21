
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f879cdcbd90> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f879cdcbd90> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f8807b129d8> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f8807b129d8> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f8825e9ce18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f8825e9ce18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f87b53b72f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f87b53b72f0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f87b53b72f0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:13, 133741.80it/s] 70%|██████▉   | 6938624/9912422 [00:00<00:15, 190901.60it/s]9920512it [00:00, 39974870.82it/s]                           
0it [00:00, ?it/s]32768it [00:00, 536435.06it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 485452.17it/s]1654784it [00:00, 11697740.48it/s]                         
0it [00:00, ?it/s]8192it [00:00, 198257.09it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87b2788cf8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87b278da20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f87b53abea0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f87b53abea0> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f87b53abea0> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:42,  1.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:41,  1.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:41,  1.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:40,  1.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:39,  1.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<01:10,  2.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:10,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:10,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:09,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:08,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:07,  2.21 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:47,  3.12 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:47,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:47,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:47,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:46,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:46,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:46,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:45,  3.12 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:32,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:32,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:32,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:31,  4.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:01<00:22,  6.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:22,  6.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:22,  6.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:21,  6.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:21,  6.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:21,  6.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:21,  6.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:21,  6.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:01<00:15,  8.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:15,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:15,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:15,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:14,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:14,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:14,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:14,  8.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 11.34 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.20 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 19.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.60 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 44.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:02<00:01, 44.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 50.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 50.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 55.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 55.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.63 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.44 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 67.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 67.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.61 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 70.61 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 72.55 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.86s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.86s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.86s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 77.20 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.86s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 77.20 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.34 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.17s/ url]
0 examples [00:00, ? examples/s]2020-06-21 00:09:50.642465: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-21 00:09:50.656261: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-06-21 00:09:50.656490: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5643aff2a570 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-21 00:09:50.656513: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  3.04 examples/s]94 examples [00:00,  4.33 examples/s]180 examples [00:00,  6.18 examples/s]269 examples [00:00,  8.80 examples/s]346 examples [00:00, 12.51 examples/s]439 examples [00:00, 17.77 examples/s]538 examples [00:00, 25.19 examples/s]637 examples [00:01, 35.60 examples/s]737 examples [00:01, 50.09 examples/s]830 examples [00:01, 69.93 examples/s]929 examples [00:01, 96.96 examples/s]1031 examples [00:01, 133.06 examples/s]1130 examples [00:01, 179.71 examples/s]1227 examples [00:01, 237.61 examples/s]1328 examples [00:01, 308.19 examples/s]1426 examples [00:01, 387.57 examples/s]1528 examples [00:01, 475.56 examples/s]1627 examples [00:02, 558.32 examples/s]1725 examples [00:02, 640.29 examples/s]1823 examples [00:02, 706.60 examples/s]1920 examples [00:02, 755.30 examples/s]2015 examples [00:02, 798.93 examples/s]2109 examples [00:02, 834.33 examples/s]2203 examples [00:02, 856.16 examples/s]2302 examples [00:02, 891.39 examples/s]2397 examples [00:02, 902.86 examples/s]2497 examples [00:02, 928.09 examples/s]2596 examples [00:03, 944.60 examples/s]2697 examples [00:03, 960.73 examples/s]2799 examples [00:03, 975.05 examples/s]2898 examples [00:03, 974.76 examples/s]2999 examples [00:03, 983.00 examples/s]3098 examples [00:03, 948.28 examples/s]3198 examples [00:03, 961.91 examples/s]3300 examples [00:03, 976.26 examples/s]3399 examples [00:03, 942.37 examples/s]3496 examples [00:03, 950.43 examples/s]3592 examples [00:04, 949.29 examples/s]3690 examples [00:04, 958.20 examples/s]3792 examples [00:04, 973.72 examples/s]3892 examples [00:04, 980.41 examples/s]3991 examples [00:04, 981.58 examples/s]4090 examples [00:04, 964.70 examples/s]4187 examples [00:04, 950.34 examples/s]4283 examples [00:04, 938.62 examples/s]4383 examples [00:04, 953.70 examples/s]4482 examples [00:05, 961.66 examples/s]4579 examples [00:05, 959.13 examples/s]4676 examples [00:05, 952.55 examples/s]4772 examples [00:05, 952.04 examples/s]4870 examples [00:05, 957.95 examples/s]4966 examples [00:05, 954.46 examples/s]5062 examples [00:05, 954.98 examples/s]5160 examples [00:05, 962.00 examples/s]5258 examples [00:05, 965.91 examples/s]5355 examples [00:05, 960.83 examples/s]5452 examples [00:06, 960.72 examples/s]5553 examples [00:06, 973.82 examples/s]5655 examples [00:06, 984.66 examples/s]5756 examples [00:06, 992.03 examples/s]5856 examples [00:06, 992.52 examples/s]5958 examples [00:06, 998.26 examples/s]6058 examples [00:06, 946.54 examples/s]6157 examples [00:06, 955.65 examples/s]6259 examples [00:06, 971.65 examples/s]6360 examples [00:06, 982.21 examples/s]6460 examples [00:07, 986.97 examples/s]6559 examples [00:07, 935.54 examples/s]6654 examples [00:07, 927.82 examples/s]6748 examples [00:07, 899.61 examples/s]6839 examples [00:07, 899.13 examples/s]6930 examples [00:07, 900.05 examples/s]7022 examples [00:07, 905.55 examples/s]7121 examples [00:07, 929.00 examples/s]7220 examples [00:07, 944.00 examples/s]7318 examples [00:07, 953.69 examples/s]7414 examples [00:08, 952.37 examples/s]7510 examples [00:08, 950.41 examples/s]7606 examples [00:08, 939.37 examples/s]7701 examples [00:08, 930.14 examples/s]7795 examples [00:08, 927.68 examples/s]7888 examples [00:08, 924.74 examples/s]7981 examples [00:08, 925.81 examples/s]8076 examples [00:08, 932.12 examples/s]8170 examples [00:08, 933.57 examples/s]8264 examples [00:09, 928.19 examples/s]8357 examples [00:09, 926.18 examples/s]8452 examples [00:09, 930.72 examples/s]8553 examples [00:09, 948.44 examples/s]8649 examples [00:09, 949.18 examples/s]8744 examples [00:09, 930.71 examples/s]8838 examples [00:09, 929.63 examples/s]8932 examples [00:09, 910.05 examples/s]9030 examples [00:09, 927.61 examples/s]9129 examples [00:09, 944.22 examples/s]9229 examples [00:10, 957.42 examples/s]9332 examples [00:10, 977.18 examples/s]9434 examples [00:10, 987.45 examples/s]9534 examples [00:10, 989.88 examples/s]9634 examples [00:10, 992.84 examples/s]9736 examples [00:10, 1000.45 examples/s]9838 examples [00:10, 1003.97 examples/s]9940 examples [00:10, 1008.55 examples/s]10041 examples [00:10, 911.84 examples/s]10134 examples [00:10, 867.93 examples/s]10224 examples [00:11, 874.72 examples/s]10316 examples [00:11, 887.42 examples/s]10406 examples [00:11, 868.70 examples/s]10494 examples [00:11, 868.65 examples/s]10582 examples [00:11, 866.11 examples/s]10676 examples [00:11, 886.09 examples/s]10766 examples [00:11, 887.83 examples/s]10861 examples [00:11, 903.81 examples/s]10952 examples [00:11, 883.54 examples/s]11044 examples [00:12, 894.14 examples/s]11142 examples [00:12, 917.18 examples/s]11243 examples [00:12, 942.04 examples/s]11346 examples [00:12, 965.12 examples/s]11448 examples [00:12, 980.05 examples/s]11547 examples [00:12, 979.27 examples/s]11647 examples [00:12, 984.33 examples/s]11746 examples [00:12, 948.49 examples/s]11842 examples [00:12, 940.06 examples/s]11941 examples [00:12, 952.95 examples/s]12040 examples [00:13, 962.75 examples/s]12140 examples [00:13, 972.44 examples/s]12239 examples [00:13, 975.88 examples/s]12339 examples [00:13, 982.29 examples/s]12438 examples [00:13, 967.41 examples/s]12535 examples [00:13, 947.76 examples/s]12630 examples [00:13, 923.34 examples/s]12725 examples [00:13, 929.10 examples/s]12826 examples [00:13, 949.53 examples/s]12927 examples [00:13, 964.37 examples/s]13031 examples [00:14, 983.38 examples/s]13133 examples [00:14, 992.73 examples/s]13233 examples [00:14, 993.66 examples/s]13335 examples [00:14, 1000.50 examples/s]13436 examples [00:14, 995.40 examples/s] 13537 examples [00:14, 999.05 examples/s]13639 examples [00:14, 1002.71 examples/s]13740 examples [00:14, 1002.97 examples/s]13841 examples [00:14, 1004.97 examples/s]13942 examples [00:14, 998.74 examples/s] 14043 examples [00:15, 1001.81 examples/s]14144 examples [00:15, 995.22 examples/s] 14244 examples [00:15, 973.69 examples/s]14348 examples [00:15, 992.37 examples/s]14449 examples [00:15, 996.30 examples/s]14552 examples [00:15, 1004.18 examples/s]14656 examples [00:15, 1012.73 examples/s]14758 examples [00:15, 1001.35 examples/s]14859 examples [00:15, 1003.22 examples/s]14960 examples [00:15, 1004.94 examples/s]15062 examples [00:16, 1007.75 examples/s]15163 examples [00:16, 1002.77 examples/s]15264 examples [00:16, 977.25 examples/s] 15362 examples [00:16, 968.37 examples/s]15459 examples [00:16, 952.46 examples/s]15556 examples [00:16, 956.42 examples/s]15652 examples [00:16, 948.67 examples/s]15747 examples [00:16, 945.68 examples/s]15851 examples [00:16, 971.36 examples/s]15952 examples [00:17, 980.49 examples/s]16056 examples [00:17, 997.25 examples/s]16160 examples [00:17, 1007.33 examples/s]16261 examples [00:17, 998.76 examples/s] 16362 examples [00:17, 973.98 examples/s]16462 examples [00:17, 980.67 examples/s]16564 examples [00:17, 989.89 examples/s]16664 examples [00:17, 989.19 examples/s]16764 examples [00:17, 971.80 examples/s]16862 examples [00:17, 953.52 examples/s]16958 examples [00:18, 942.89 examples/s]17057 examples [00:18, 955.86 examples/s]17159 examples [00:18, 972.15 examples/s]17259 examples [00:18, 978.41 examples/s]17357 examples [00:18, 940.40 examples/s]17452 examples [00:18, 939.83 examples/s]17550 examples [00:18, 949.12 examples/s]17651 examples [00:18, 965.80 examples/s]17748 examples [00:18, 938.87 examples/s]17849 examples [00:18, 959.07 examples/s]17953 examples [00:19, 980.31 examples/s]18052 examples [00:19, 969.75 examples/s]18156 examples [00:19, 987.47 examples/s]18259 examples [00:19, 998.49 examples/s]18360 examples [00:19, 995.23 examples/s]18460 examples [00:19, 991.95 examples/s]18560 examples [00:19, 988.82 examples/s]18661 examples [00:19, 992.77 examples/s]18763 examples [00:19, 1000.75 examples/s]18866 examples [00:19, 1005.39 examples/s]18967 examples [00:20, 949.17 examples/s] 19063 examples [00:20, 950.96 examples/s]19161 examples [00:20, 958.29 examples/s]19263 examples [00:20, 974.28 examples/s]19363 examples [00:20, 981.53 examples/s]19462 examples [00:20, 980.95 examples/s]19565 examples [00:20, 992.64 examples/s]19667 examples [00:20, 999.79 examples/s]19768 examples [00:20, 988.35 examples/s]19867 examples [00:21, 972.83 examples/s]19965 examples [00:21, 959.67 examples/s]20062 examples [00:21, 920.57 examples/s]20165 examples [00:21, 950.79 examples/s]20268 examples [00:21, 971.49 examples/s]20371 examples [00:21, 985.21 examples/s]20470 examples [00:21, 971.38 examples/s]20572 examples [00:21, 982.89 examples/s]20672 examples [00:21, 985.36 examples/s]20771 examples [00:21, 972.56 examples/s]20872 examples [00:22, 978.19 examples/s]20973 examples [00:22, 987.07 examples/s]21075 examples [00:22, 995.58 examples/s]21177 examples [00:22, 1001.70 examples/s]21279 examples [00:22, 1006.01 examples/s]21380 examples [00:22, 994.81 examples/s] 21482 examples [00:22, 1001.95 examples/s]21586 examples [00:22, 1012.96 examples/s]21690 examples [00:22, 1018.93 examples/s]21793 examples [00:22, 1020.11 examples/s]21896 examples [00:23, 1015.97 examples/s]21998 examples [00:23, 1004.66 examples/s]22099 examples [00:23, 1006.06 examples/s]22200 examples [00:23, 991.31 examples/s] 22300 examples [00:23, 992.14 examples/s]22400 examples [00:23, 977.74 examples/s]22498 examples [00:23, 971.53 examples/s]22600 examples [00:23, 984.40 examples/s]22700 examples [00:23, 987.33 examples/s]22799 examples [00:23, 986.55 examples/s]22898 examples [00:24, 972.74 examples/s]22996 examples [00:24, 929.07 examples/s]23090 examples [00:24, 914.57 examples/s]23182 examples [00:24, 900.03 examples/s]23279 examples [00:24, 919.02 examples/s]23374 examples [00:24, 927.74 examples/s]23468 examples [00:24, 922.08 examples/s]23561 examples [00:24, 902.68 examples/s]23652 examples [00:24, 870.16 examples/s]23740 examples [00:25, 855.94 examples/s]23838 examples [00:25, 888.40 examples/s]23938 examples [00:25, 916.60 examples/s]24033 examples [00:25, 923.33 examples/s]24128 examples [00:25, 930.91 examples/s]24227 examples [00:25, 943.61 examples/s]24325 examples [00:25, 952.43 examples/s]24421 examples [00:25, 944.55 examples/s]24518 examples [00:25, 951.35 examples/s]24617 examples [00:25, 960.34 examples/s]24714 examples [00:26, 920.28 examples/s]24807 examples [00:26, 920.49 examples/s]24900 examples [00:26, 923.32 examples/s]25000 examples [00:26, 943.68 examples/s]25095 examples [00:26, 912.18 examples/s]25187 examples [00:26, 903.69 examples/s]25287 examples [00:26, 930.44 examples/s]25381 examples [00:26, 932.34 examples/s]25482 examples [00:26, 952.86 examples/s]25583 examples [00:27, 967.40 examples/s]25681 examples [00:27, 969.77 examples/s]25781 examples [00:27, 977.08 examples/s]25881 examples [00:27, 982.11 examples/s]25983 examples [00:27, 988.71 examples/s]26085 examples [00:27, 997.63 examples/s]26185 examples [00:27, 989.95 examples/s]26286 examples [00:27, 995.72 examples/s]26386 examples [00:27, 985.82 examples/s]26489 examples [00:27, 996.37 examples/s]26589 examples [00:28, 955.90 examples/s]26685 examples [00:28, 922.24 examples/s]26778 examples [00:28, 917.88 examples/s]26871 examples [00:28, 914.18 examples/s]26965 examples [00:28, 918.32 examples/s]27065 examples [00:28, 939.43 examples/s]27160 examples [00:28, 915.16 examples/s]27260 examples [00:28, 937.04 examples/s]27361 examples [00:28, 955.10 examples/s]27461 examples [00:28, 967.90 examples/s]27563 examples [00:29, 981.07 examples/s]27662 examples [00:29, 967.90 examples/s]27759 examples [00:29, 963.45 examples/s]27856 examples [00:29, 962.41 examples/s]27953 examples [00:29, 950.20 examples/s]28049 examples [00:29, 946.62 examples/s]28144 examples [00:29, 913.53 examples/s]28240 examples [00:29, 925.44 examples/s]28338 examples [00:29, 941.14 examples/s]28438 examples [00:29, 956.63 examples/s]28535 examples [00:30, 956.24 examples/s]28638 examples [00:30, 975.09 examples/s]28736 examples [00:30, 968.25 examples/s]28840 examples [00:30, 988.44 examples/s]28944 examples [00:30, 1002.42 examples/s]29045 examples [00:30, 989.03 examples/s] 29148 examples [00:30, 998.53 examples/s]29248 examples [00:30, 986.40 examples/s]29347 examples [00:30, 970.67 examples/s]29445 examples [00:31, 935.70 examples/s]29542 examples [00:31, 943.76 examples/s]29644 examples [00:31, 961.65 examples/s]29741 examples [00:31, 942.95 examples/s]29841 examples [00:31, 957.41 examples/s]29942 examples [00:31, 971.76 examples/s]30040 examples [00:31, 914.16 examples/s]30133 examples [00:31, 916.38 examples/s]30226 examples [00:31, 913.97 examples/s]30318 examples [00:31, 915.08 examples/s]30415 examples [00:32, 927.88 examples/s]30511 examples [00:32, 935.71 examples/s]30613 examples [00:32, 957.70 examples/s]30715 examples [00:32, 975.28 examples/s]30817 examples [00:32, 986.82 examples/s]30916 examples [00:32, 980.46 examples/s]31015 examples [00:32, 970.88 examples/s]31114 examples [00:32, 975.53 examples/s]31212 examples [00:32, 972.92 examples/s]31310 examples [00:32, 968.33 examples/s]31411 examples [00:33, 978.38 examples/s]31509 examples [00:33, 963.99 examples/s]31607 examples [00:33, 968.32 examples/s]31704 examples [00:33, 968.10 examples/s]31801 examples [00:33, 942.61 examples/s]31896 examples [00:33, 917.83 examples/s]31989 examples [00:33, 908.74 examples/s]32083 examples [00:33, 916.11 examples/s]32179 examples [00:33, 926.20 examples/s]32280 examples [00:34, 949.38 examples/s]32376 examples [00:34, 936.76 examples/s]32475 examples [00:34, 949.92 examples/s]32574 examples [00:34, 961.25 examples/s]32678 examples [00:34, 982.82 examples/s]32778 examples [00:34, 986.17 examples/s]32877 examples [00:34, 967.13 examples/s]32974 examples [00:34, 943.49 examples/s]33069 examples [00:34, 933.00 examples/s]33163 examples [00:34, 924.85 examples/s]33258 examples [00:35, 931.33 examples/s]33357 examples [00:35, 946.65 examples/s]33455 examples [00:35, 955.37 examples/s]33559 examples [00:35, 977.96 examples/s]33660 examples [00:35, 985.40 examples/s]33759 examples [00:35, 973.03 examples/s]33857 examples [00:35, 970.79 examples/s]33955 examples [00:35, 960.69 examples/s]34058 examples [00:35, 979.01 examples/s]34162 examples [00:35, 992.72 examples/s]34267 examples [00:36, 1007.42 examples/s]34368 examples [00:36, 997.63 examples/s] 34469 examples [00:36, 999.46 examples/s]34574 examples [00:36, 1013.25 examples/s]34677 examples [00:36, 1016.22 examples/s]34782 examples [00:36, 1024.37 examples/s]34885 examples [00:36, 1011.36 examples/s]34987 examples [00:36, 974.39 examples/s] 35087 examples [00:36, 980.03 examples/s]35186 examples [00:36, 979.41 examples/s]35285 examples [00:37, 979.28 examples/s]35384 examples [00:37, 937.86 examples/s]35479 examples [00:37, 936.09 examples/s]35574 examples [00:37, 939.86 examples/s]35676 examples [00:37, 958.89 examples/s]35779 examples [00:37, 977.56 examples/s]35878 examples [00:37, 968.71 examples/s]35976 examples [00:37, 957.86 examples/s]36073 examples [00:37, 960.69 examples/s]36172 examples [00:38, 967.77 examples/s]36271 examples [00:38, 972.79 examples/s]36371 examples [00:38, 978.06 examples/s]36471 examples [00:38, 983.26 examples/s]36573 examples [00:38, 993.49 examples/s]36675 examples [00:38, 1000.15 examples/s]36776 examples [00:38, 999.53 examples/s] 36876 examples [00:38, 989.46 examples/s]36975 examples [00:38, 972.07 examples/s]37073 examples [00:38, 962.23 examples/s]37170 examples [00:39, 954.64 examples/s]37271 examples [00:39, 968.20 examples/s]37368 examples [00:39, 924.85 examples/s]37461 examples [00:39, 903.53 examples/s]37562 examples [00:39, 931.49 examples/s]37667 examples [00:39, 961.82 examples/s]37769 examples [00:39, 978.01 examples/s]37871 examples [00:39, 988.22 examples/s]37975 examples [00:39, 1001.27 examples/s]38076 examples [00:39, 1003.39 examples/s]38181 examples [00:40, 1015.79 examples/s]38283 examples [00:40, 977.87 examples/s] 38382 examples [00:40, 956.10 examples/s]38479 examples [00:40, 949.61 examples/s]38575 examples [00:40, 936.29 examples/s]38672 examples [00:40, 943.71 examples/s]38767 examples [00:40, 943.49 examples/s]38867 examples [00:40, 959.67 examples/s]38964 examples [00:40, 876.30 examples/s]39054 examples [00:41, 838.41 examples/s]39144 examples [00:41, 854.20 examples/s]39243 examples [00:41, 889.39 examples/s]39340 examples [00:41, 912.04 examples/s]39440 examples [00:41, 935.42 examples/s]39542 examples [00:41, 956.89 examples/s]39639 examples [00:41, 957.74 examples/s]39736 examples [00:41, 952.79 examples/s]39832 examples [00:41, 950.61 examples/s]39933 examples [00:41, 965.16 examples/s]40030 examples [00:42, 932.93 examples/s]40128 examples [00:42, 945.40 examples/s]40226 examples [00:42, 954.93 examples/s]40326 examples [00:42, 966.31 examples/s]40427 examples [00:42, 979.00 examples/s]40526 examples [00:42, 946.11 examples/s]40621 examples [00:42, 940.46 examples/s]40717 examples [00:42, 945.93 examples/s]40818 examples [00:42, 963.45 examples/s]40917 examples [00:42, 970.91 examples/s]41015 examples [00:43, 966.30 examples/s]41112 examples [00:43, 955.48 examples/s]41208 examples [00:43, 950.16 examples/s]41308 examples [00:43, 963.93 examples/s]41405 examples [00:43, 965.04 examples/s]41502 examples [00:43, 943.51 examples/s]41597 examples [00:43, 944.58 examples/s]41692 examples [00:43, 941.16 examples/s]41793 examples [00:43, 958.49 examples/s]41894 examples [00:44, 971.86 examples/s]41996 examples [00:44, 983.19 examples/s]42095 examples [00:44, 961.55 examples/s]42195 examples [00:44, 970.89 examples/s]42293 examples [00:44, 972.03 examples/s]42392 examples [00:44, 977.19 examples/s]42493 examples [00:44, 986.30 examples/s]42592 examples [00:44, 982.17 examples/s]42692 examples [00:44, 986.41 examples/s]42793 examples [00:44, 993.08 examples/s]42893 examples [00:45, 991.33 examples/s]42993 examples [00:45, 991.51 examples/s]43093 examples [00:45, 981.16 examples/s]43192 examples [00:45, 978.74 examples/s]43290 examples [00:45, 978.82 examples/s]43391 examples [00:45, 986.36 examples/s]43492 examples [00:45, 992.03 examples/s]43592 examples [00:45, 992.80 examples/s]43692 examples [00:45, 984.83 examples/s]43794 examples [00:45, 993.02 examples/s]43895 examples [00:46, 997.74 examples/s]43997 examples [00:46, 1002.13 examples/s]44098 examples [00:46, 971.84 examples/s] 44198 examples [00:46, 977.72 examples/s]44300 examples [00:46, 989.00 examples/s]44404 examples [00:46, 1001.28 examples/s]44505 examples [00:46, 987.95 examples/s] 44605 examples [00:46, 990.72 examples/s]44706 examples [00:46, 995.35 examples/s]44806 examples [00:46, 984.19 examples/s]44906 examples [00:47, 987.19 examples/s]45005 examples [00:47, 979.81 examples/s]45104 examples [00:47, 975.62 examples/s]45202 examples [00:47, 957.16 examples/s]45302 examples [00:47, 969.34 examples/s]45403 examples [00:47, 979.50 examples/s]45502 examples [00:47, 971.83 examples/s]45600 examples [00:47, 972.14 examples/s]45698 examples [00:47, 973.00 examples/s]45797 examples [00:47, 976.22 examples/s]45899 examples [00:48, 988.53 examples/s]46001 examples [00:48, 996.55 examples/s]46102 examples [00:48, 998.73 examples/s]46204 examples [00:48, 1004.38 examples/s]46305 examples [00:48, 993.18 examples/s] 46407 examples [00:48, 999.06 examples/s]46508 examples [00:48, 1001.66 examples/s]46610 examples [00:48, 1004.18 examples/s]46711 examples [00:48, 928.67 examples/s] 46806 examples [00:49, 925.96 examples/s]46906 examples [00:49, 946.71 examples/s]47006 examples [00:49, 960.03 examples/s]47103 examples [00:49, 937.81 examples/s]47205 examples [00:49, 958.70 examples/s]47307 examples [00:49, 973.92 examples/s]47408 examples [00:49, 983.74 examples/s]47507 examples [00:49, 984.48 examples/s]47607 examples [00:49, 986.58 examples/s]47706 examples [00:49, 978.64 examples/s]47804 examples [00:50, 965.06 examples/s]47901 examples [00:50, 962.04 examples/s]48004 examples [00:50, 979.61 examples/s]48106 examples [00:50, 989.24 examples/s]48209 examples [00:50, 1000.48 examples/s]48310 examples [00:50, 998.74 examples/s] 48410 examples [00:50, 984.11 examples/s]48511 examples [00:50, 991.67 examples/s]48612 examples [00:50, 993.72 examples/s]48712 examples [00:50, 988.54 examples/s]48814 examples [00:51, 996.65 examples/s]48918 examples [00:51, 1008.15 examples/s]49022 examples [00:51, 1015.39 examples/s]49125 examples [00:51, 1018.89 examples/s]49227 examples [00:51, 1016.19 examples/s]49329 examples [00:51, 1017.13 examples/s]49433 examples [00:51, 1014.41 examples/s]49535 examples [00:51, 1010.61 examples/s]49637 examples [00:51, 1008.59 examples/s]49738 examples [00:51, 996.36 examples/s] 49838 examples [00:52, 989.52 examples/s]49937 examples [00:52, 976.86 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6328/50000 [00:00<00:00, 63274.48 examples/s] 36%|███▌      | 17856/50000 [00:00<00:00, 73177.15 examples/s] 57%|█████▋    | 28749/50000 [00:00<00:00, 81167.67 examples/s] 78%|███████▊  | 38976/50000 [00:00<00:00, 86522.35 examples/s]                                                               0 examples [00:00, ? examples/s]83 examples [00:00, 822.78 examples/s]178 examples [00:00, 855.84 examples/s]275 examples [00:00, 885.83 examples/s]370 examples [00:00, 903.93 examples/s]475 examples [00:00, 941.11 examples/s]575 examples [00:00, 956.86 examples/s]673 examples [00:00, 963.53 examples/s]773 examples [00:00, 971.65 examples/s]872 examples [00:00, 975.15 examples/s]971 examples [00:01, 978.50 examples/s]1067 examples [00:01, 966.19 examples/s]1163 examples [00:01, 941.86 examples/s]1257 examples [00:01, 927.98 examples/s]1350 examples [00:01, 920.56 examples/s]1451 examples [00:01, 944.46 examples/s]1554 examples [00:01, 968.34 examples/s]1655 examples [00:01, 979.45 examples/s]1757 examples [00:01, 991.22 examples/s]1859 examples [00:01, 998.48 examples/s]1962 examples [00:02, 1006.27 examples/s]2066 examples [00:02, 1013.88 examples/s]2168 examples [00:02, 985.09 examples/s] 2267 examples [00:02, 971.76 examples/s]2365 examples [00:02, 966.02 examples/s]2462 examples [00:02, 960.60 examples/s]2559 examples [00:02, 928.65 examples/s]2658 examples [00:02, 944.12 examples/s]2757 examples [00:02, 955.42 examples/s]2853 examples [00:02, 949.51 examples/s]2951 examples [00:03, 957.56 examples/s]3053 examples [00:03, 974.16 examples/s]3154 examples [00:03, 984.44 examples/s]3257 examples [00:03, 996.15 examples/s]3360 examples [00:03, 1003.79 examples/s]3464 examples [00:03, 1012.32 examples/s]3566 examples [00:03, 1013.40 examples/s]3668 examples [00:03, 1011.13 examples/s]3771 examples [00:03, 1014.93 examples/s]3873 examples [00:03, 998.39 examples/s] 3975 examples [00:04, 1004.56 examples/s]4078 examples [00:04, 1010.31 examples/s]4180 examples [00:04, 976.13 examples/s] 4278 examples [00:04, 971.73 examples/s]4376 examples [00:04, 964.47 examples/s]4481 examples [00:04, 987.40 examples/s]4580 examples [00:04, 980.49 examples/s]4679 examples [00:04, 965.58 examples/s]4782 examples [00:04, 982.64 examples/s]4884 examples [00:05, 993.31 examples/s]4989 examples [00:05, 1007.50 examples/s]5093 examples [00:05, 1015.06 examples/s]5195 examples [00:05, 1013.66 examples/s]5297 examples [00:05, 1011.90 examples/s]5399 examples [00:05, 1010.51 examples/s]5501 examples [00:05, 1012.54 examples/s]5603 examples [00:05, 980.10 examples/s] 5704 examples [00:05, 986.58 examples/s]5803 examples [00:05, 975.99 examples/s]5901 examples [00:06, 965.76 examples/s]5998 examples [00:06, 953.18 examples/s]6094 examples [00:06, 942.85 examples/s]6189 examples [00:06, 930.44 examples/s]6283 examples [00:06, 920.24 examples/s]6384 examples [00:06, 943.52 examples/s]6486 examples [00:06, 963.56 examples/s]6587 examples [00:06, 975.06 examples/s]6685 examples [00:06, 974.72 examples/s]6785 examples [00:06, 980.42 examples/s]6886 examples [00:07, 989.10 examples/s]6989 examples [00:07, 1000.56 examples/s]7091 examples [00:07, 1003.87 examples/s]7192 examples [00:07, 991.97 examples/s] 7292 examples [00:07, 977.67 examples/s]7390 examples [00:07, 976.71 examples/s]7488 examples [00:07, 973.81 examples/s]7586 examples [00:07, 970.93 examples/s]7685 examples [00:07, 976.17 examples/s]7783 examples [00:07, 953.73 examples/s]7879 examples [00:08, 948.25 examples/s]7977 examples [00:08, 955.41 examples/s]8075 examples [00:08, 960.98 examples/s]8172 examples [00:08, 963.34 examples/s]8270 examples [00:08, 966.00 examples/s]8371 examples [00:08, 977.20 examples/s]8471 examples [00:08, 983.37 examples/s]8570 examples [00:08, 944.28 examples/s]8667 examples [00:08, 950.56 examples/s]8763 examples [00:08, 945.05 examples/s]8858 examples [00:09, 917.51 examples/s]8958 examples [00:09, 938.54 examples/s]9061 examples [00:09, 963.17 examples/s]9161 examples [00:09, 972.70 examples/s]9261 examples [00:09, 977.92 examples/s]9364 examples [00:09, 992.00 examples/s]9465 examples [00:09, 996.05 examples/s]9568 examples [00:09, 1003.86 examples/s]9669 examples [00:09, 1001.27 examples/s]9770 examples [00:10, 999.90 examples/s] 9872 examples [00:10, 1003.14 examples/s]9973 examples [00:10, 1004.57 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLCC4LU/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteLCC4LU/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f87b53b72f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f87b53b72f0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f87b53b72f0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f88193b62b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f8750081470>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f87b53b72f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f87b53b72f0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f87b53b72f0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87b27a82e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f87b27a8048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f872ce6b1e0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f872ce6b1e0> 

  function with postional parmater data_info <function split_train_valid at 0x7f872ce6b1e0> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f872ce6b2f0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f872ce6b2f0> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f872ce6b2f0> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.19.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.6.20)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=ecd3d1f5fb52957d5f17f420f2d523a0638c4ad7780c71715cde9214e3b3aae2
  Stored in directory: /tmp/pip-ephem-wheel-cache-vgjlmzji/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:33:47, 12.2kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<13:55:40, 17.2kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:48:13, 24.4kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<6:52:16, 34.8kB/s].vector_cache/glove.6B.zip:   0%|          | 2.92M/862M [00:01<4:48:07, 49.7kB/s].vector_cache/glove.6B.zip:   1%|          | 6.64M/862M [00:01<3:20:56, 71.0kB/s].vector_cache/glove.6B.zip:   1%|▏         | 10.8M/862M [00:01<2:20:04, 101kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.4M/862M [00:01<1:37:37, 145kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.8M/862M [00:01<1:08:10, 206kB/s].vector_cache/glove.6B.zip:   3%|▎         | 23.5M/862M [00:01<47:33, 294kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 27.4M/862M [00:01<33:14, 419kB/s].vector_cache/glove.6B.zip:   4%|▎         | 31.7M/862M [00:01<23:14, 595kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.7M/862M [00:01<16:17, 845kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:02<11:26, 1.20MB/s].vector_cache/glove.6B.zip:   5%|▌         | 43.5M/862M [00:02<08:05, 1.69MB/s].vector_cache/glove.6B.zip:   6%|▌         | 48.3M/862M [00:02<05:43, 2.37MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.9M/862M [00:02<04:24, 3.07MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.1M/862M [00:02<03:14, 4.15MB/s].vector_cache/glove.6B.zip:   6%|▋         | 56.0M/862M [00:04<10:25, 1.29MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<13:53, 967kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 56.4M/862M [00:04<11:29, 1.17MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.6M/862M [00:05<08:25, 1.59MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<06:05, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:06<15:48, 846kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.3M/862M [00:06<14:18, 934kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:06<10:49, 1.23MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.1M/862M [00:06<07:44, 1.72MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.3M/862M [00:08<10:57, 1.21MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.5M/862M [00:08<10:55, 1.22MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:08<08:20, 1.59MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.0M/862M [00:08<06:02, 2.20MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<08:40, 1.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.6M/862M [00:10<09:08, 1.45MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.3M/862M [00:10<07:03, 1.87MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.7M/862M [00:10<05:13, 2.53MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:11<04:42, 2.80MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:12<8:14:45, 26.6kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<5:46:45, 37.9kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:12<4:02:05, 54.2kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:14<2:58:19, 73.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<2:14:04, 97.7kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.6M/862M [00:14<1:36:05, 136kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 77.6M/862M [00:14<1:07:44, 193kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:14<47:25, 275kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:16<45:40, 285kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.6M/862M [00:16<35:57, 362kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<26:09, 498kB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.1M/862M [00:16<18:30, 701kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<17:09, 756kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:18<15:46, 822kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.3M/862M [00:18<12:00, 1.08MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.3M/862M [00:18<08:37, 1.50MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:20<10:18, 1.25MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.9M/862M [00:20<12:54, 998kB/s] .vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<10:24, 1.24MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:20<07:36, 1.69MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:20<05:34, 2.30MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:22<49:02, 261kB/s] .vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<40:05, 320kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.4M/862M [00:22<29:24, 436kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<20:53, 612kB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:22<14:49, 860kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:24<1:34:37, 135kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:24<1:11:27, 178kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.6M/862M [00:24<51:16, 249kB/s]  .vector_cache/glove.6B.zip:  11%|█▏        | 99.1M/862M [00:24<36:09, 352kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<28:04, 452kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<24:48, 511kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<18:23, 689kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<13:10, 961kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:26<09:31, 1.33MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<12:20, 1.02MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<13:47, 915kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<10:53, 1.16MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<07:52, 1.60MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:28<05:46, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<11:51, 1.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<12:58, 967kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<10:04, 1.24MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<07:18, 1.71MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<05:42, 2.19MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<08:12, 1.52MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:30, 997kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<12:36, 989kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<09:54, 1.26MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<08:29, 1.47MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<07:24, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:24, 1.94MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<06:30, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<06:01, 2.06MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<05:33, 2.24MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 116M/862M [00:33<05:07, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:22, 2.31MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<05:02, 2.46MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<04:44, 2.62MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:01, 2.47MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<09:27, 1.31MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<08:05, 1.53MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<07:02, 1.76MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<06:16, 1.97MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:43, 2.16MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:20, 2.32MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:05, 2.43MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:54, 2.52MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:44, 2.61MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<04:36, 2.68MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:34, 2.70MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:30, 2.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:29, 2.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:29, 2.74MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<04:29, 2.75MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<13:13, 933kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<10:54, 1.13MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<08:45, 1.41MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<07:49, 1.58MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<06:44, 1.83MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:58, 2.06MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:24, 2.27MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<05:02, 2.44MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<04:45, 2.58MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<04:31, 2.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<04:31, 2.71MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<04:22, 2.81MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<04:20, 2.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<23:43, 517kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<18:32, 662kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<14:14, 860kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<11:11, 1.09MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<08:55, 1.37MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<07:26, 1.65MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<06:22, 1.92MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:38<05:37, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<05:10, 2.36MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:42, 2.59MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<04:25, 2.76MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<21:53, 557kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<18:20, 665kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<14:06, 864kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<10:59, 1.11MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<08:48, 1.38MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<07:15, 1.68MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<06:09, 1.97MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:40<05:23, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<04:51, 2.50MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<04:27, 2.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<03:57, 3.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<4:55:49, 41.0kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<3:29:53, 57.8kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<2:27:59, 81.9kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<1:44:34, 116kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<1:14:12, 163kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<52:50, 229kB/s]  .vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<38:04, 318kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<27:28, 440kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:42<21:07, 572kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<15:37, 773kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<12:14, 985kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<19:29, 619kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<17:20, 696kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<13:49, 872kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<11:38, 1.04MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:51, 1.22MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<08:57, 1.34MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:56, 1.51MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:24, 1.63MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<07:01, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:41, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:26, 1.87MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:11, 1.94MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<06:07, 1.96MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<06:00, 2.00MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:58, 2.01MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:56, 2.02MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<05:56, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:45<05:44, 2.09MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:47, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:38, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:48, 2.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:38, 2.12MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<05:49, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:35, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:45, 2.08MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:30, 2.17MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:37, 2.13MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:24, 2.21MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:48, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:25, 2.20MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<05:39, 2.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:17, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:28, 2.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:13, 2.29MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:17, 2.26MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<05:11, 2.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<04:57, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:13, 2.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<04:58, 2.40MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<05:05, 2.34MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:54, 2.43MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<05:00, 2.38MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:45, 2.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<04:56, 2.41MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:46, 2.49MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:45, 2.50MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<04:43, 2.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:44, 2.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:41, 2.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<11:05, 1.07MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<09:07, 1.30MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<07:35, 1.56MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:49<06:49, 1.74MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:57, 1.99MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<05:42, 2.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:15, 2.25MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:01, 2.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<04:47, 2.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:41, 2.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:30, 2.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:26, 2.66MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<04:21, 2.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:20, 2.72MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<04:21, 2.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<59:39, 198kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<45:34, 259kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:51<33:11, 355kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<24:26, 482kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<18:20, 643kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<13:47, 854kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:15, 1.05MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:48, 1.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<07:46, 1.51MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<06:21, 1.85MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:59, 1.96MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<05:19, 2.20MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:37, 2.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<04:52, 2.41MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<33:35, 349kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<25:13, 465kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<18:48, 623kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<14:16, 821kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<11:05, 1.06MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:53<08:36, 1.36MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:20, 1.59MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:16, 1.86MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<05:10, 2.26MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<04:59, 2.34MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:13, 2.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<08:58, 1.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<09:49, 1.19MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<08:11, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:47, 1.72MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:55<06:10, 1.88MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<05:23, 2.16MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<05:15, 2.21MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<04:54, 2.37MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:31, 2.57MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:24, 2.64MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:04, 2.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<04:05, 2.84MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:52, 3.00MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<03:49, 3.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<19:32, 593kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<15:14, 760kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:57<11:43, 989kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<09:09, 1.27MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<07:11, 1.61MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<06:28, 1.79MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:57<05:31, 2.09MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:50, 2.38MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<04:22, 2.64MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<04:07, 2.80MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:48, 3.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<03:41, 3.13MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<27:14, 423kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<21:39, 532kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<16:17, 707kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<12:22, 930kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<09:36, 1.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 172M/862M [00:59<07:39, 1.50MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:19, 1.82MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<05:30, 2.09MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:47, 2.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:18, 2.66MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<03:56, 2.91MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<19:17, 594kB/s] .vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<16:02, 714kB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<12:16, 932kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<09:32, 1.20MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<07:36, 1.50MB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:01<06:15, 1.83MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:01<05:17, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:39, 2.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:11, 2.72MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<03:51, 2.95MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<37:54, 300kB/s] .vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<29:03, 392kB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:03<21:24, 532kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<15:54, 715kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<12:04, 942kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<09:21, 1.21MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<07:28, 1.52MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<06:10, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<05:15, 2.16MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:04<04:37, 2.45MB/s].vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<27:29, 412kB/s] .vector_cache/glove.6B.zip:  21%|██        | 183M/862M [01:05<22:42, 499kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:05<17:11, 658kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<13:18, 850kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<10:31, 1.07MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<08:29, 1.33MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<07:11, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<06:07, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<05:27, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<04:51, 2.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:30, 2.50MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:10, 2.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<04:03, 2.78MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:06<03:48, 2.95MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<11:01, 1.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:07<09:14, 1.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<07:31, 1.49MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<06:26, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:07<05:47, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:27, 2.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<05:09, 2.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<04:59, 2.25MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:38, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:46, 2.34MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:33, 2.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<04:37, 2.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:21, 2.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<04:26, 2.52MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<14:10, 789kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:09<11:41, 957kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<09:23, 1.19MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<07:47, 1.43MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:09<06:27, 1.73MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:51, 1.90MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<05:18, 2.10MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<04:53, 2.28MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:09<04:37, 2.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<04:22, 2.55MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<04:13, 2.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<04:17, 2.59MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:10, 2.67MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:05, 2.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<10:02, 1.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:11<08:44, 1.27MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<07:18, 1.52MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:11<06:16, 1.77MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<05:32, 2.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:49, 2.30MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<04:53, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:21, 2.54MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:11<04:25, 2.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<04:18, 2.56MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<03:54, 2.83MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:12, 2.63MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<04:04, 2.71MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<14:04, 785kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<11:28, 962kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<09:07, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<07:31, 1.46MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:13<06:18, 1.75MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:31, 2.00MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:53, 2.26MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<04:31, 2.44MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:12, 2.61MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:13<04:00, 2.74MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<03:49, 2.88MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:44, 2.94MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:36, 3.05MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<03:35, 3.06MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<33:20, 329kB/s] .vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<26:00, 422kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<19:22, 566kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:15<14:35, 751kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<11:04, 990kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<08:43, 1.25MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<07:07, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<06:13, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<05:17, 2.07MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:43, 2.31MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:15, 2.56MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<04:00, 2.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<03:56, 2.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<16:58, 643kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<14:33, 749kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<11:08, 979kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:17<09:02, 1.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<07:05, 1.54MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:11, 1.76MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<05:19, 2.05MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:38, 2.34MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<04:19, 2.52MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<03:55, 2.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:17<03:40, 2.95MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<03:28, 3.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:18<03:21, 3.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<14:11, 764kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<12:20, 878kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:19<09:39, 1.12MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<07:39, 1.41MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<06:17, 1.72MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<05:13, 2.07MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<04:35, 2.35MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<03:59, 2.71MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:39, 2.95MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<03:20, 3.22MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 216M/862M [01:19<03:04, 3.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<12:55, 833kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<11:08, 967kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:21<08:35, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:21<06:53, 1.56MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:23, 1.99MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:18, 2.49MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<04:28, 2.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:57, 2.71MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:21<03:33, 3.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<03:10, 3.38MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<19:12, 557kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:22<16:44, 639kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:23<12:37, 847kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:23<09:26, 1.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<07:06, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:37, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<04:38, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:23<03:41, 2.89MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:15, 3.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<44:10, 241kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<33:38, 316kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:25<24:21, 436kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<17:33, 604kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<12:40, 836kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:25<09:27, 1.12MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<07:07, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<12:00, 879kB/s] .vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:26<13:08, 804kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<10:12, 1.04MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<07:35, 1.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<05:37, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:30, 2.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<07:02, 1.49MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<08:52, 1.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:15, 1.44MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<05:28, 1.91MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<04:12, 2.48MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<03:17, 3.18MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<02:36, 4.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<2:46:43, 62.6kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<1:59:51, 87.0kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<1:24:38, 123kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<59:28, 175kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<43:04, 241kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:31<30:16, 343kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:31<21:37, 479kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<2:18:42, 74.7kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:41:17, 102kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<1:11:54, 144kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<50:38, 204kB/s]  .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<35:45, 289kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:33<25:17, 407kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<29:18, 351kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<24:22, 422kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<17:53, 575kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<12:53, 797kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<09:22, 1.09MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:35<06:53, 1.49MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<05:12, 1.96MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<41:26, 247kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<32:24, 315kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<23:34, 433kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<16:44, 609kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<12:11, 835kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<08:51, 1.15MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<11:08, 911kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<11:04, 916kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<08:28, 1.20MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<06:16, 1.61MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:41, 2.15MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:37, 2.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<12:09, 830kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<5:15:52, 31.9kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<3:42:42, 45.3kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<2:36:05, 64.5kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:40<1:49:27, 91.9kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:40<1:16:45, 131kB/s] .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:40<53:58, 186kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<1:01:39, 163kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<53:06, 189kB/s]  .vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<39:17, 255kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<28:01, 357kB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<20:06, 497kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:42<14:26, 691kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:42<10:30, 950kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<07:43, 1.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<13:53, 716kB/s] .vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<13:39, 729kB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<10:29, 948kB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<07:52, 1.26MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<05:51, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<04:30, 2.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:44<03:28, 2.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<08:25, 1.17MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<09:57, 992kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<07:46, 1.27MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<06:01, 1.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<04:51, 2.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:46<03:41, 2.67MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:46<03:05, 3.18MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<02:39, 3.68MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<1:01:25, 160kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<48:19, 203kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<35:11, 279kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:48<25:05, 390kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<18:00, 543kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<13:03, 748kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<09:35, 1.02MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<14:16, 682kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<15:21, 635kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:50<12:01, 810kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<08:49, 1.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<06:36, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:50<05:03, 1.92MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:50<03:58, 2.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<03:14, 2.99MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<36:12, 267kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<30:04, 322kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<22:16, 434kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<16:02, 602kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<11:41, 825kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:52<08:46, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<06:44, 1.43MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<05:17, 1.82MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<26:23, 364kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<21:36, 444kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<15:55, 603kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<11:42, 819kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<08:44, 1.10MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<06:33, 1.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [01:54<05:20, 1.79MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:15, 2.25MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<10:46, 885kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<10:28, 911kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<08:05, 1.18MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<06:04, 1.57MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:57, 1.92MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:56<04:18, 2.20MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:26, 2.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:56<03:02, 3.12MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<02:56, 3.23MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<11:13, 843kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<11:46, 804kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<09:01, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<07:15, 1.30MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:58<05:36, 1.68MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<04:52, 1.94MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:58, 2.37MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<03:35, 2.62MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:58<03:03, 3.07MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<02:58, 3.17MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<02:36, 3.62MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<11:28, 819kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<11:32, 814kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<09:00, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<06:57, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<05:29, 1.71MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<04:53, 1.92MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:00<04:01, 2.33MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<03:40, 2.55MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<03:24, 2.74MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<03:13, 2.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<18:25, 507kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<15:04, 619kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<11:23, 819kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<08:47, 1.06MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<06:56, 1.34MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<05:38, 1.65MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:42, 1.98MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<04:02, 2.29MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<03:35, 2.59MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<03:20, 2.78MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<16:26, 564kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<14:20, 646kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<12:05, 766kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<09:22, 987kB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:04<07:50, 1.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<07:07, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<06:14, 1.48MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:44, 1.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<05:40, 1.63MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:05<05:18, 1.74MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:05, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:51, 1.90MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<04:43, 1.95MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:37, 1.99MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:32, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:32, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:26, 2.07MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:06<04:21, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:20, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:18, 2.13MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:12, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:13, 2.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:10, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:10, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:10, 2.20MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<04:10, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:07<04:07, 2.22MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:08, 2.21MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:06, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:04, 2.25MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<04:01, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:01, 2.27MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:56, 2.32MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:05, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<03:52, 2.35MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<04:00, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:48, 2.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:00, 2.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<03:47, 2.40MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:49, 2.39MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:43, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:43, 2.44MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:08<03:41, 2.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:40, 2.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:39, 2.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:45, 2.42MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<03:40, 2.47MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:46, 2.41MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<03:44, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:53, 1.15MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<07:03, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:59, 1.51MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<05:13, 1.73MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:40, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:17, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<04:00, 2.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:48, 2.37MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:38, 2.48MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:33, 2.53MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:10<03:26, 2.62MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:10<03:23, 2.66MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:17, 2.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<03:17, 2.74MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<13:33, 664kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<11:09, 807kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<08:59, 1.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<07:16, 1.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<06:04, 1.48MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:12<05:07, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:34, 1.96MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:03, 2.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<03:50, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:34, 2.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:26, 2.60MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:12<03:12, 2.79MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:13, 2.78MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<03:03, 2.92MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<39:17, 228kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<1:05:51, 136kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<46:57, 190kB/s]  .vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<33:42, 265kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<24:15, 368kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<18:09, 491kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<13:40, 651kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<10:54, 817kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:14<08:25, 1.06MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<06:43, 1.32MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<05:28, 1.63MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<04:40, 1.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<04:01, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:14<03:32, 2.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<20:39, 429kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<16:26, 539kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<12:16, 721kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<09:18, 950kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<07:10, 1.23MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<05:41, 1.55MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:16<04:34, 1.93MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:51, 2.29MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<03:19, 2.66MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<02:53, 3.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:16<02:36, 3.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<33:28, 263kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<26:33, 331kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<19:27, 452kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<14:04, 624kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<10:21, 846kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<07:42, 1.14MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<05:54, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:39, 1.88MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:36, 2.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<5:39:53, 25.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<4:00:10, 36.3kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<2:48:34, 51.7kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<1:58:20, 73.6kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<1:23:09, 105kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<58:31, 148kB/s]  .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<41:17, 210kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<29:15, 296kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<32:55, 263kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<26:58, 321kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<19:50, 436kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<14:20, 603kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:21<10:19, 836kB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<07:35, 1.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:21<05:34, 1.54MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<16:09, 532kB/s] .vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<14:32, 591kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<10:58, 782kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<07:54, 1.08MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<05:56, 1.44MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:23<04:24, 1.94MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<09:16, 920kB/s] .vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<09:04, 939kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<07:00, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<05:08, 1.65MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<03:48, 2.23MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:25<02:56, 2.88MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<1:13:11, 116kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<55:52, 151kB/s]  .vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<40:14, 210kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<28:27, 296kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<20:03, 419kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:27<14:13, 590kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<24:27, 343kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<20:53, 401kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<15:37, 536kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<11:22, 736kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<08:14, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<05:57, 1.40MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<07:26, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<06:57, 1.19MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<05:18, 1.56MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:50, 2.15MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:53, 2.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<21:03, 392kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<17:57, 459kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<13:19, 618kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<09:39, 852kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<06:55, 1.18MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<07:15, 1.13MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<07:44, 1.06MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:06, 1.34MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<04:26, 1.83MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:35<03:14, 2.51MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<46:25, 175kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<34:58, 232kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<25:04, 323kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<17:38, 457kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<14:36, 550kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<12:25, 647kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<09:22, 857kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<06:49, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<04:54, 1.63MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:44, 1.18MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<06:40, 1.19MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<05:03, 1.57MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:48, 2.08MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:12, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 388M/862M [02:43<04:49, 1.64MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:46, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<02:57, 2.66MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:43<02:10, 3.59MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<06:52, 1.14MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<06:49, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:16, 1.48MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:48, 2.05MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:04, 1.53MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<05:38, 1.38MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<04:25, 1.75MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:47<03:13, 2.39MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<04:17, 1.79MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:49<04:47, 1.60MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<03:45, 2.04MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<02:47, 2.74MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:49, 2.00MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<04:05, 1.87MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<03:32, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:39, 2.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<03:28, 2.17MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<04:09, 1.81MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<03:46, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<02:55, 2.57MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<02:09, 3.48MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<04:57, 1.51MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<05:35, 1.34MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:49, 1.55MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:36, 2.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:37, 2.82MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<06:09, 1.21MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<06:01, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [02:57<04:35, 1.61MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<03:22, 2.19MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<04:11, 1.75MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:05, 1.79MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<03:12, 2.28MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:22, 3.08MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:54, 1.87MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:00<03:36, 2.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<02:47, 2.59MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:04, 3.50MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:23, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<04:24, 1.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:25, 2.10MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<02:30, 2.85MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:29, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<04:28, 1.60MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:24, 2.08MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:05<02:29, 2.85MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:45, 1.49MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:39, 1.52MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:06<03:34, 1.97MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:39, 2.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:37, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<03:51, 1.81MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<02:58, 2.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:10, 3.20MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:14, 1.32MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:00, 1.39MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<03:49, 1.81MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:45, 2.49MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:30, 1.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:07, 1.34MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<03:51, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:12<02:46, 2.46MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<06:51, 990kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<07:20, 925kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<05:40, 1.20MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<04:08, 1.63MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:15<02:58, 2.26MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<13:10, 510kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<10:29, 641kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<07:38, 878kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<06:27, 1.03MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<07:02, 945kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<05:27, 1.22MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<04:05, 1.62MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:56, 1.67MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:00, 1.65MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:06, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:20<02:13, 2.93MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<06:22, 1.02MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<4:34:03, 23.8kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<3:12:14, 33.9kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<2:14:08, 48.4kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<1:35:01, 67.9kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<1:10:03, 92.1kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<49:54, 129kB/s]   .vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<35:03, 183kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<25:31, 250kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<19:41, 324kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<14:06, 452kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<10:05, 631kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:26<07:06, 890kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<10:48, 584kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<09:13, 685kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<06:45, 933kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<04:56, 1.27MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:39, 1.34MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:30<04:50, 1.29MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<03:46, 1.66MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<02:49, 2.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:30<02:02, 3.02MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<28:17, 218kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<22:12, 278kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<16:03, 384kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<11:31, 535kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:32<08:06, 755kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<08:29, 720kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:34<08:09, 749kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<06:16, 974kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:34<04:39, 1.31MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<03:21, 1.80MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:18, 1.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<05:08, 1.18MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:06, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:59, 2.01MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<03:40, 1.62MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<04:40, 1.28MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<03:46, 1.58MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:44, 2.16MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:38<02:01, 2.93MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<30:42, 192kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<23:33, 250kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<16:53, 349kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<11:58, 491kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:40<08:25, 694kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<11:15, 518kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<09:46, 596kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<07:20, 793kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:42<05:21, 1.08MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<03:53, 1.49MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:42<02:49, 2.04MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<11:30, 501kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<10:12, 564kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<07:36, 756kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<05:26, 1.05MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:44<03:53, 1.47MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<06:46, 841kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<06:31, 872kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<05:30, 1.03MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<04:08, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<03:03, 1.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<02:15, 2.50MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:30, 1.25MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<05:16, 1.07MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<04:13, 1.33MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<03:03, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:48<02:14, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:39, 982kB/s] .vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<05:46, 963kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<04:42, 1.18MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<03:33, 1.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:45, 2.00MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:28, 2.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<01:58, 2.79MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:49, 3.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<01:41, 3.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<06:32, 839kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<06:50, 803kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<05:20, 1.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<04:07, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:16, 1.67MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:36, 2.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:17, 2.37MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:52<01:58, 2.75MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<01:46, 3.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:39, 3.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:58, 1.09MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<05:31, 981kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<04:23, 1.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:21, 1.61MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:45, 1.95MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:16, 2.36MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:53, 2.85MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:48, 2.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<01:35, 3.35MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<07:52, 680kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<07:26, 719kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<05:59, 893kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<04:45, 1.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<03:44, 1.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:18, 1.61MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:38, 2.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<02:26, 2.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:12, 2.39MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:07, 2.49MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<02:10, 2.43MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:03, 2.58MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<01:56, 2.73MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<01:46, 2.96MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<09:07, 578kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<07:04, 746kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<05:25, 972kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<04:11, 1.26MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<03:23, 1.55MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:46, 1.90MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:22, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<02:06, 2.49MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:57, 2.68MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<01:47, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:40, 3.13MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<01:35, 3.29MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<07:49, 666kB/s] .vector_cache/glove.6B.zip:  64%|██████▎   | 550M/862M [04:00<06:35, 791kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<05:02, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:55, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<03:08, 1.65MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:32, 2.03MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:10, 2.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:00<01:54, 2.70MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<01:41, 3.05MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<01:36, 3.19MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:01<54:00, 95.2kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<39:53, 129kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<28:28, 180kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<20:14, 253kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<14:40, 349kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<10:58, 466kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<07:58, 640kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<06:02, 843kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:02<04:50, 1.05MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:48, 1.34MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:17, 1.55MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:43, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<02:28, 2.06MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:03<08:31, 595kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<07:14, 701kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<05:32, 915kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<04:23, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:36, 1.40MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<03:02, 1.66MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:39, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:20, 2.16MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:02, 2.47MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:52, 2.68MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:43, 2.90MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<01:36, 3.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<01:32, 3.24MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<04:51, 1.03MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<04:28, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:33, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:50, 1.76MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:25, 2.05MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:06, 2.36MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:53, 2.64MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<01:45, 2.83MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:35, 3.12MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:29, 3.31MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:06<01:25, 3.49MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<04:45, 1.04MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<04:20, 1.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<03:27, 1.42MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<02:47, 1.76MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:19, 2.11MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<01:59, 2.46MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:47, 2.73MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:37, 3.00MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:08<01:32, 3.16MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:08<01:25, 3.43MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<05:36, 868kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<05:55, 822kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<04:45, 1.02MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:10<03:38, 1.33MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:10<03:03, 1.59MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:25, 1.99MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:11, 2.20MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:51, 2.59MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:43, 2.79MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:10<01:31, 3.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:10<01:28, 3.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:23, 3.46MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<14:27, 332kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<11:05, 433kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<08:12, 583kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<06:03, 789kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<04:35, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<03:34, 1.33MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:53, 1.65MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:12<02:20, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:57, 2.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:12<01:49, 2.60MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<17:24, 272kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<14:09, 334kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<10:21, 456kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<07:47, 606kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<05:46, 816kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<04:25, 1.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<03:24, 1.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:46, 1.70MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<02:15, 2.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:58, 2.38MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:14<01:40, 2.80MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:14<01:34, 2.96MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<07:58, 585kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<06:28, 719kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<04:53, 949kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<03:45, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:58, 1.55MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:24, 1.92MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<02:00, 2.30MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:43, 2.69MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:31, 3.03MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:22, 3.36MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<06:09, 745kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<05:59, 766kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<04:47, 956kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<03:42, 1.24MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:55, 1.57MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<02:21, 1.94MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:57, 2.32MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<01:38, 2.76MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:27, 3.13MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:18<01:17, 3.52MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:18<01:11, 3.78MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<10:13, 443kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<08:37, 525kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<06:26, 702kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<04:42, 956kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<03:39, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:45, 1.63MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:16, 1.98MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:47, 2.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:20<01:34, 2.84MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:44, 941kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<04:37, 963kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:34, 1.24MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:42, 1.64MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:08, 2.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<01:43, 2.57MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:23, 3.17MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:22<01:12, 3.61MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<07:17, 602kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<07:15, 604kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<05:34, 786kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:23<04:04, 1.07MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:06, 1.40MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:22, 1.83MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<01:50, 2.36MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<01:29, 2.91MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:10, 1.03MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<04:50, 892kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<03:50, 1.12MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<02:51, 1.51MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<02:09, 1.99MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<01:40, 2.54MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:26<01:17, 3.30MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<29:17, 145kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<22:05, 192kB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:27<15:54, 267kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<11:24, 371kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:27<08:08, 519kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<05:49, 723kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<04:11, 1.00MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<03:55, 1.07MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<2:11:24, 31.8kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<1:32:34, 45.1kB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<1:04:45, 64.3kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<45:10, 91.7kB/s]  .vector_cache/glove.6B.zip:  71%|███████▏  | 614M/862M [04:29<31:41, 130kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<24:13, 170kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<19:12, 214kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<14:05, 292kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<10:08, 405kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<07:10, 570kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:31<05:06, 796kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<05:01, 804kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<04:34, 883kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<03:27, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<02:30, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:33<01:48, 2.20MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<55:04, 72.2kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<40:25, 98.3kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<28:45, 138kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<20:20, 195kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:35<14:17, 276kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:35<10:01, 391kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<08:59, 435kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<07:52, 496kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<05:51, 667kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<04:15, 914kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:37<03:01, 1.28MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:32, 1.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:55, 979kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:06, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<02:15, 1.68MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:24, 1.57MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<02:55, 1.29MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:21, 1.59MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:45, 2.12MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:41<01:17, 2.89MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:24, 1.09MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<03:30, 1.05MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<02:41, 1.37MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 642M/862M [04:43<01:56, 1.89MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:43<01:26, 2.52MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<04:12, 863kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<04:08, 876kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<03:08, 1.15MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:18, 1.57MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:45<01:38, 2.17MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:59, 714kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<04:27, 798kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<03:20, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<02:22, 1.48MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:55, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:54, 1.20MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<02:13, 1.56MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:36, 2.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:03, 1.66MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<02:22, 1.44MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<01:53, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<01:23, 2.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:51<01:02, 3.21MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<10:40, 314kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<09:17, 361kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<06:56, 483kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<04:55, 674kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<03:29, 942kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:13, 778kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<04:33, 720kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<03:35, 911kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<02:35, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:54, 1.71MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:55<01:23, 2.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<11:57, 269kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<09:48, 328kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<07:13, 444kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<05:07, 622kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<03:36, 874kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<05:07, 614kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<04:59, 630kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<03:50, 818kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<02:45, 1.13MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:58, 1.56MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:26, 692kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<04:24, 697kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<03:24, 900kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<02:27, 1.24MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<01:45, 1.71MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:51, 620kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<04:37, 650kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<03:29, 861kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:29, 1.20MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:50, 1.61MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:15, 1.30MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<03:11, 920kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:36, 1.13MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<02:02, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:30, 1.93MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:07, 2.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:05<00:51, 3.36MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<23:43, 121kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<17:22, 165kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:07<12:23, 231kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<08:48, 324kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<06:12, 457kB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<04:25, 638kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<04:09, 675kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:09<03:37, 773kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<02:42, 1.03MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:55, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<01:26, 1.90MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:15, 1.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:04, 890kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:29, 1.09MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:48, 1.50MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:24, 1.91MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:11<01:01, 2.59MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<03:33, 747kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:13<03:31, 754kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:43, 973kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<02:00, 1.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:29, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:52, 3.00MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<05:06, 507kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<04:29, 578kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<03:19, 778kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<02:22, 1.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<01:45, 1.46MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:15<01:18, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<02:09, 1.17MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<03:02, 828kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<02:34, 981kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:52, 1.33MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<01:23, 1.80MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:17<01:03, 2.35MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:47, 1.38MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<02:03, 1.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:37, 1.50MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:15, 1.93MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:55, 2.59MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:19<00:44, 3.24MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:56, 810kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<03:07, 765kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<02:24, 991kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:21<01:48, 1.31MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:21, 1.74MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<01:02, 2.25MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:21<00:49, 2.84MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:21<00:39, 3.51MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<04:47, 483kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<04:19, 535kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<03:15, 708kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<02:21, 971kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<01:43, 1.32MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:23<01:17, 1.76MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:30, 900kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<02:39, 845kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:05, 1.07MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:34, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:09, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:54, 2.43MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<00:46, 2.80MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:52, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:30, 870kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<02:00, 1.08MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<01:30, 1.43MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<01:10, 1.83MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:54, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:42, 2.98MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:39, 3.25MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<09:55, 213kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<07:58, 265kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<05:48, 363kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<04:08, 505kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<02:57, 703kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<02:13, 933kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:37, 1.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<01:13, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<04:58, 410kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<04:28, 456kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:30<03:22, 603kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<02:26, 826kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:49, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<01:20, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:31<01:03, 1.87MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:31<00:52, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<03:16, 602kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:58, 661kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:15, 873kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<01:41, 1.16MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<01:17, 1.51MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<01:00, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:48, 2.37MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:33<00:40, 2.84MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<06:01, 316kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<04:46, 398kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<03:29, 544kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<02:31, 744kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<01:51, 1.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:22, 1.35MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<01:05, 1.69MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:35<00:51, 2.13MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:02, 902kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:59, 924kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:36<01:31, 1.20MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:09, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:53, 2.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:45, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:37, 2.89MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:37<00:32, 3.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:31, 1.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:35, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<01:14, 1.41MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<00:57, 1.81MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:45, 2.27MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:39<00:37, 2.78MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:39<00:31, 3.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:37, 1.04MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:32, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:20, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:02, 1.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:40<00:49, 2.05MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:39, 2.53MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:32, 3.02MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:28, 3.49MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:25, 3.89MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:41<00:50, 1.94MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<41:07, 39.7kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<28:56, 56.2kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:42<20:13, 80.0kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<14:07, 114kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<09:53, 161kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<06:57, 227kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<04:54, 319kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<06:58, 224kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<05:49, 268kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<04:18, 361kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<03:09, 491kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:17, 675kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:44<01:41, 909kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<01:15, 1.21MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:57, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 772M/862M [05:45<00:44, 2.02MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:36, 2.50MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:56, 768kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:48, 828kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:22, 1.08MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:02, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:46<00:47, 1.84MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:37, 2.31MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:47<00:30, 2.82MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:31, 938kB/s] .vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:50, 775kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:28, 959kB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<01:05, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:48<00:49, 1.70MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:39, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:32, 2.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:26, 3.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:49<00:23, 3.47MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:46, 766kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:36, 841kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<01:12, 1.12MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:53, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:41, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:32, 2.39MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:50<00:26, 2.94MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:22, 3.50MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:35, 810kB/s] .vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:44, 740kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<01:22, 927kB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<01:01, 1.24MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:52<00:45, 1.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:34, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:28, 2.56MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:53<00:23, 3.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<07:30, 163kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<05:50, 208kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<04:12, 288kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<02:58, 403kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:54<02:06, 563kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<01:30, 775kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:54<01:05, 1.06MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:29, 775kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:30, 762kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<01:10, 972kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:51, 1.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:56<00:38, 1.75MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:28, 2.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:56<00:24, 2.67MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:41, 642kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<01:50, 589kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:25, 755kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<01:02, 1.02MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:58<00:46, 1.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:34, 1.79MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:26, 2.28MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<01:09, 873kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:18, 775kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:02, 972kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:45, 1.31MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:37, 1.58MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:00<00:27, 2.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:00<00:22, 2.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:19, 2.89MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<01:28, 636kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<01:02, 890kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:46, 1.19MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:02<00:34, 1.55MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:02<00:27, 1.97MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:21, 2.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:03<00:17, 2.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<02:57, 296kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<02:19, 377kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:40, 516kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<01:12, 709kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:53, 954kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:39, 1.27MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:04<00:30, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:23, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:04<00:19, 2.49MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<01:00, 795kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:56, 856kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:42, 1.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:31, 1.47MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:24, 1.90MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:06<00:19, 2.39MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:06<00:15, 2.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:51, 863kB/s] .vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:59, 737kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:47, 918kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:35, 1.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:26, 1.59MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:20, 2.04MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:16, 2.52MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:08<00:13, 2.98MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:08<00:11, 3.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:51, 784kB/s] .vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:46, 859kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:34, 1.14MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:26, 1.48MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:10<00:19, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:15, 2.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:12, 2.90MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:10, 3.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<01:03, 564kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:12<01:04, 560kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:49, 720kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:35, 973kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:26, 1.31MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:20, 1.67MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:12<00:15, 2.16MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:12<00:12, 2.55MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:10, 2.99MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:12, 442kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:14<00:58, 540kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:45, 698kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:32, 941kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:24, 1.25MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:14<00:18, 1.63MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:14, 2.09MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:14<00:11, 2.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 3.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:32, 858kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:30, 917kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:16<00:22, 1.19MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:16, 1.58MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:12, 1.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:16<00:10, 2.46MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:08, 2.99MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:16<00:07, 3.39MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:21, 1.10MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:20, 1.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:18<00:15, 1.45MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:11, 1.89MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:09, 2.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:18<00:07, 2.90MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:05, 3.43MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:18<00:05, 3.85MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:43, 450kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:39, 487kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:20<00:29, 642kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:21, 872kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<00:15, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:20<00:10, 1.55MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:08, 2.02MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<00:06, 2.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:15, 958kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:18, 818kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:22<00:14, 1.04MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:10, 1.39MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:07, 1.79MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:05, 2.31MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.88MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:03, 3.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:10, 1.07MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:12, 875kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:24<00:09, 1.11MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:06, 1.49MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:04, 1.90MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:03, 2.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:24<00:02, 3.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:24<00:02, 3.59MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:09, 764kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:09, 756kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:06, 969kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:04, 1.30MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:26<00:02, 1.75MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.28MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:01, 2.75MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:08, 365kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:07, 395kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:04, 527kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:02, 724kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:01, 992kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:28<00:00, 1.27MB/s].vector_cache/glove.6B.zip: 862MB [06:28, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<18:58:16,  5.86it/s]  0%|          | 652/400000 [00:00<13:15:47,  8.36it/s]  0%|          | 1329/400000 [00:00<9:16:24, 11.94it/s]  0%|          | 1973/400000 [00:00<6:29:10, 17.05it/s]  1%|          | 2651/400000 [00:00<4:32:14, 24.33it/s]  1%|          | 3331/400000 [00:00<3:10:32, 34.70it/s]  1%|          | 4025/400000 [00:00<2:13:25, 49.46it/s]  1%|          | 4753/400000 [00:00<1:33:30, 70.45it/s]  1%|▏         | 5475/400000 [00:00<1:05:36, 100.23it/s]  2%|▏         | 6237/400000 [00:01<46:05, 142.38it/s]    2%|▏         | 6990/400000 [00:01<32:27, 201.77it/s]  2%|▏         | 7714/400000 [00:01<22:57, 284.83it/s]  2%|▏         | 8429/400000 [00:01<16:19, 399.92it/s]  2%|▏         | 9139/400000 [00:01<11:41, 557.49it/s]  2%|▏         | 9873/400000 [00:01<08:25, 771.29it/s]  3%|▎         | 10627/400000 [00:01<06:08, 1055.54it/s]  3%|▎         | 11387/400000 [00:01<04:33, 1423.19it/s]  3%|▎         | 12123/400000 [00:01<03:27, 1871.26it/s]  3%|▎         | 12850/400000 [00:01<02:41, 2394.87it/s]  3%|▎         | 13575/400000 [00:02<02:08, 2996.52it/s]  4%|▎         | 14294/400000 [00:02<01:47, 3596.30it/s]  4%|▍         | 15013/400000 [00:02<01:30, 4230.66it/s]  4%|▍         | 15737/400000 [00:02<01:19, 4832.35it/s]  4%|▍         | 16459/400000 [00:02<01:11, 5363.13it/s]  4%|▍         | 17220/400000 [00:02<01:05, 5883.32it/s]  4%|▍         | 17950/400000 [00:02<01:01, 6244.60it/s]  5%|▍         | 18680/400000 [00:02<00:59, 6453.50it/s]  5%|▍         | 19428/400000 [00:02<00:56, 6730.03it/s]  5%|▌         | 20173/400000 [00:02<00:54, 6930.37it/s]  5%|▌         | 20934/400000 [00:03<00:53, 7120.07it/s]  5%|▌         | 21677/400000 [00:03<00:52, 7208.89it/s]  6%|▌         | 22419/400000 [00:03<00:52, 7224.30it/s]  6%|▌         | 23185/400000 [00:03<00:51, 7347.94it/s]  6%|▌         | 23931/400000 [00:03<00:51, 7269.58it/s]  6%|▌         | 24670/400000 [00:03<00:51, 7305.18it/s]  6%|▋         | 25406/400000 [00:03<00:52, 7149.11it/s]  7%|▋         | 26126/400000 [00:03<00:53, 7011.90it/s]  7%|▋         | 26831/400000 [00:03<00:54, 6803.52it/s]  7%|▋         | 27555/400000 [00:04<00:53, 6928.53it/s]  7%|▋         | 28308/400000 [00:04<00:52, 7096.53it/s]  7%|▋         | 29031/400000 [00:04<00:51, 7134.64it/s]  7%|▋         | 29785/400000 [00:04<00:51, 7250.94it/s]  8%|▊         | 30535/400000 [00:04<00:50, 7323.20it/s]  8%|▊         | 31269/400000 [00:04<00:50, 7263.79it/s]  8%|▊         | 32014/400000 [00:04<00:50, 7317.38it/s]  8%|▊         | 32748/400000 [00:04<00:50, 7321.15it/s]  8%|▊         | 33490/400000 [00:04<00:49, 7348.96it/s]  9%|▊         | 34226/400000 [00:04<00:49, 7341.73it/s]  9%|▊         | 34961/400000 [00:05<00:49, 7309.08it/s]  9%|▉         | 35693/400000 [00:05<00:49, 7308.48it/s]  9%|▉         | 36425/400000 [00:05<00:49, 7288.66it/s]  9%|▉         | 37173/400000 [00:05<00:49, 7344.86it/s]  9%|▉         | 37908/400000 [00:05<00:49, 7312.42it/s] 10%|▉         | 38640/400000 [00:05<00:51, 7046.16it/s] 10%|▉         | 39377/400000 [00:05<00:50, 7140.22it/s] 10%|█         | 40093/400000 [00:05<00:50, 7140.22it/s] 10%|█         | 40818/400000 [00:05<00:50, 7172.56it/s] 10%|█         | 41546/400000 [00:05<00:49, 7203.59it/s] 11%|█         | 42268/400000 [00:06<00:49, 7198.60it/s] 11%|█         | 43035/400000 [00:06<00:48, 7331.22it/s] 11%|█         | 43781/400000 [00:06<00:48, 7369.35it/s] 11%|█         | 44547/400000 [00:06<00:47, 7453.95it/s] 11%|█▏        | 45294/400000 [00:06<00:47, 7435.11it/s] 12%|█▏        | 46039/400000 [00:06<00:49, 7190.68it/s] 12%|█▏        | 46797/400000 [00:06<00:48, 7302.65it/s] 12%|█▏        | 47530/400000 [00:06<00:48, 7210.47it/s] 12%|█▏        | 48253/400000 [00:06<00:48, 7208.46it/s] 12%|█▏        | 49008/400000 [00:06<00:48, 7304.77it/s] 12%|█▏        | 49740/400000 [00:07<00:48, 7291.01it/s] 13%|█▎        | 50478/400000 [00:07<00:47, 7317.38it/s] 13%|█▎        | 51237/400000 [00:07<00:47, 7395.18it/s] 13%|█▎        | 51978/400000 [00:07<00:48, 7110.00it/s] 13%|█▎        | 52698/400000 [00:07<00:48, 7134.69it/s] 13%|█▎        | 53414/400000 [00:07<00:48, 7107.53it/s] 14%|█▎        | 54145/400000 [00:07<00:48, 7166.55it/s] 14%|█▎        | 54886/400000 [00:07<00:47, 7236.57it/s] 14%|█▍        | 55611/400000 [00:07<00:47, 7231.67it/s] 14%|█▍        | 56335/400000 [00:07<00:47, 7174.24it/s] 14%|█▍        | 57053/400000 [00:08<00:48, 7140.06it/s] 14%|█▍        | 57774/400000 [00:08<00:47, 7160.39it/s] 15%|█▍        | 58494/400000 [00:08<00:47, 7171.84it/s] 15%|█▍        | 59212/400000 [00:08<00:48, 7038.30it/s] 15%|█▍        | 59946/400000 [00:08<00:47, 7120.34it/s] 15%|█▌        | 60659/400000 [00:08<00:47, 7122.96it/s] 15%|█▌        | 61372/400000 [00:08<00:47, 7084.05it/s] 16%|█▌        | 62109/400000 [00:08<00:47, 7165.75it/s] 16%|█▌        | 62867/400000 [00:08<00:46, 7284.30it/s] 16%|█▌        | 63610/400000 [00:09<00:45, 7326.28it/s] 16%|█▌        | 64344/400000 [00:09<00:45, 7319.94it/s] 16%|█▋        | 65077/400000 [00:09<00:47, 7121.61it/s] 16%|█▋        | 65791/400000 [00:09<00:47, 7105.27it/s] 17%|█▋        | 66515/400000 [00:09<00:46, 7144.24it/s] 17%|█▋        | 67231/400000 [00:09<00:46, 7134.32it/s] 17%|█▋        | 67971/400000 [00:09<00:46, 7211.33it/s] 17%|█▋        | 68704/400000 [00:09<00:45, 7244.36it/s] 17%|█▋        | 69451/400000 [00:09<00:45, 7309.41it/s] 18%|█▊        | 70183/400000 [00:09<00:45, 7182.93it/s] 18%|█▊        | 70903/400000 [00:10<00:45, 7186.57it/s] 18%|█▊        | 71623/400000 [00:10<00:46, 7000.36it/s] 18%|█▊        | 72325/400000 [00:10<00:48, 6826.01it/s] 18%|█▊        | 73061/400000 [00:10<00:46, 6975.71it/s] 18%|█▊        | 73781/400000 [00:10<00:46, 7039.22it/s] 19%|█▊        | 74494/400000 [00:10<00:46, 7065.84it/s] 19%|█▉        | 75229/400000 [00:10<00:45, 7146.86it/s] 19%|█▉        | 75967/400000 [00:10<00:44, 7214.67it/s] 19%|█▉        | 76690/400000 [00:10<00:45, 7145.54it/s] 19%|█▉        | 77421/400000 [00:10<00:44, 7192.78it/s] 20%|█▉        | 78141/400000 [00:11<00:45, 7023.32it/s] 20%|█▉        | 78881/400000 [00:11<00:45, 7130.34it/s] 20%|█▉        | 79597/400000 [00:11<00:44, 7137.64it/s] 20%|██        | 80346/400000 [00:11<00:44, 7238.58it/s] 20%|██        | 81086/400000 [00:11<00:43, 7284.67it/s] 20%|██        | 81816/400000 [00:11<00:44, 7184.47it/s] 21%|██        | 82536/400000 [00:11<00:47, 6664.39it/s] 21%|██        | 83243/400000 [00:11<00:46, 6779.33it/s] 21%|██        | 83990/400000 [00:11<00:45, 6971.86it/s] 21%|██        | 84736/400000 [00:11<00:44, 7110.65it/s] 21%|██▏       | 85452/400000 [00:12<00:44, 7105.35it/s] 22%|██▏       | 86166/400000 [00:12<00:44, 7111.21it/s] 22%|██▏       | 86880/400000 [00:12<00:44, 7050.90it/s] 22%|██▏       | 87587/400000 [00:12<00:45, 6862.00it/s] 22%|██▏       | 88284/400000 [00:12<00:45, 6893.54it/s] 22%|██▏       | 88976/400000 [00:12<00:45, 6869.46it/s] 22%|██▏       | 89712/400000 [00:12<00:44, 7008.78it/s] 23%|██▎       | 90447/400000 [00:12<00:43, 7105.34it/s] 23%|██▎       | 91159/400000 [00:12<00:43, 7081.12it/s] 23%|██▎       | 91869/400000 [00:13<00:44, 6856.27it/s] 23%|██▎       | 92557/400000 [00:13<00:45, 6739.16it/s] 23%|██▎       | 93242/400000 [00:13<00:45, 6771.69it/s] 23%|██▎       | 93921/400000 [00:13<00:46, 6532.25it/s] 24%|██▎       | 94613/400000 [00:13<00:45, 6643.07it/s] 24%|██▍       | 95363/400000 [00:13<00:44, 6876.93it/s] 24%|██▍       | 96081/400000 [00:13<00:43, 6963.71it/s] 24%|██▍       | 96799/400000 [00:13<00:43, 7025.96it/s] 24%|██▍       | 97513/400000 [00:13<00:42, 7058.70it/s] 25%|██▍       | 98221/400000 [00:13<00:43, 6958.10it/s] 25%|██▍       | 98934/400000 [00:14<00:42, 7006.93it/s] 25%|██▍       | 99636/400000 [00:14<00:45, 6657.19it/s] 25%|██▌       | 100326/400000 [00:14<00:44, 6727.93it/s] 25%|██▌       | 101076/400000 [00:14<00:43, 6941.49it/s] 25%|██▌       | 101834/400000 [00:14<00:41, 7120.47it/s] 26%|██▌       | 102560/400000 [00:14<00:41, 7159.17it/s] 26%|██▌       | 103290/400000 [00:14<00:41, 7200.63it/s] 26%|██▌       | 104018/400000 [00:14<00:40, 7222.14it/s] 26%|██▌       | 104742/400000 [00:14<00:41, 7178.15it/s] 26%|██▋       | 105461/400000 [00:14<00:41, 7163.96it/s] 27%|██▋       | 106179/400000 [00:15<00:41, 7126.58it/s] 27%|██▋       | 106917/400000 [00:15<00:40, 7200.53it/s] 27%|██▋       | 107672/400000 [00:15<00:40, 7299.55it/s] 27%|██▋       | 108430/400000 [00:15<00:39, 7379.14it/s] 27%|██▋       | 109180/400000 [00:15<00:39, 7412.39it/s] 27%|██▋       | 109948/400000 [00:15<00:38, 7489.64it/s] 28%|██▊       | 110698/400000 [00:15<00:39, 7268.14it/s] 28%|██▊       | 111463/400000 [00:15<00:39, 7377.64it/s] 28%|██▊       | 112233/400000 [00:15<00:38, 7469.94it/s] 28%|██▊       | 112984/400000 [00:15<00:38, 7481.48it/s] 28%|██▊       | 113734/400000 [00:16<00:38, 7378.55it/s] 29%|██▊       | 114473/400000 [00:16<00:40, 7073.06it/s] 29%|██▉       | 115184/400000 [00:16<00:41, 6922.96it/s] 29%|██▉       | 115923/400000 [00:16<00:40, 7056.29it/s] 29%|██▉       | 116665/400000 [00:16<00:39, 7160.83it/s] 29%|██▉       | 117388/400000 [00:16<00:39, 7179.59it/s] 30%|██▉       | 118108/400000 [00:16<00:40, 6999.17it/s] 30%|██▉       | 118872/400000 [00:16<00:39, 7177.72it/s] 30%|██▉       | 119634/400000 [00:16<00:38, 7303.06it/s] 30%|███       | 120367/400000 [00:17<00:38, 7311.00it/s] 30%|███       | 121118/400000 [00:17<00:37, 7367.97it/s] 30%|███       | 121857/400000 [00:17<00:39, 6990.08it/s] 31%|███       | 122561/400000 [00:17<00:40, 6833.31it/s] 31%|███       | 123249/400000 [00:17<00:41, 6748.67it/s] 31%|███       | 124005/400000 [00:17<00:39, 6972.74it/s] 31%|███       | 124735/400000 [00:17<00:38, 7066.46it/s] 31%|███▏      | 125445/400000 [00:17<00:40, 6809.98it/s] 32%|███▏      | 126131/400000 [00:17<00:41, 6646.19it/s] 32%|███▏      | 126861/400000 [00:17<00:39, 6829.49it/s] 32%|███▏      | 127557/400000 [00:18<00:39, 6867.39it/s] 32%|███▏      | 128277/400000 [00:18<00:39, 6961.67it/s] 32%|███▏      | 128976/400000 [00:18<00:39, 6903.71it/s] 32%|███▏      | 129697/400000 [00:18<00:38, 6992.82it/s] 33%|███▎      | 130442/400000 [00:18<00:37, 7121.48it/s] 33%|███▎      | 131191/400000 [00:18<00:37, 7226.36it/s] 33%|███▎      | 131933/400000 [00:18<00:36, 7283.12it/s] 33%|███▎      | 132663/400000 [00:18<00:38, 6944.05it/s] 33%|███▎      | 133362/400000 [00:18<00:38, 6881.78it/s] 34%|███▎      | 134067/400000 [00:18<00:38, 6930.21it/s] 34%|███▎      | 134763/400000 [00:19<00:38, 6907.38it/s] 34%|███▍      | 135466/400000 [00:19<00:38, 6943.46it/s] 34%|███▍      | 136177/400000 [00:19<00:37, 6990.94it/s] 34%|███▍      | 136925/400000 [00:19<00:36, 7129.25it/s] 34%|███▍      | 137642/400000 [00:19<00:36, 7141.14it/s] 35%|███▍      | 138358/400000 [00:19<00:37, 6992.91it/s] 35%|███▍      | 139059/400000 [00:19<00:38, 6826.32it/s] 35%|███▍      | 139744/400000 [00:19<00:38, 6727.50it/s] 35%|███▌      | 140436/400000 [00:19<00:38, 6782.68it/s] 35%|███▌      | 141116/400000 [00:20<00:38, 6712.29it/s] 35%|███▌      | 141789/400000 [00:20<00:39, 6617.00it/s] 36%|███▌      | 142501/400000 [00:20<00:38, 6758.78it/s] 36%|███▌      | 143262/400000 [00:20<00:36, 6993.02it/s] 36%|███▌      | 144025/400000 [00:20<00:35, 7172.41it/s] 36%|███▌      | 144788/400000 [00:20<00:34, 7303.48it/s] 36%|███▋      | 145522/400000 [00:20<00:35, 7117.85it/s] 37%|███▋      | 146237/400000 [00:20<00:37, 6826.39it/s] 37%|███▋      | 146925/400000 [00:20<00:37, 6719.56it/s] 37%|███▋      | 147633/400000 [00:20<00:36, 6823.23it/s] 37%|███▋      | 148376/400000 [00:21<00:35, 6994.18it/s] 37%|███▋      | 149079/400000 [00:21<00:36, 6883.67it/s] 37%|███▋      | 149770/400000 [00:21<00:37, 6683.36it/s] 38%|███▊      | 150522/400000 [00:21<00:36, 6912.55it/s] 38%|███▊      | 151284/400000 [00:21<00:34, 7108.22it/s] 38%|███▊      | 152000/400000 [00:21<00:35, 7031.86it/s] 38%|███▊      | 152707/400000 [00:21<00:35, 6996.33it/s] 38%|███▊      | 153438/400000 [00:21<00:34, 7086.34it/s] 39%|███▊      | 154179/400000 [00:21<00:34, 7178.25it/s] 39%|███▊      | 154932/400000 [00:21<00:33, 7277.45it/s] 39%|███▉      | 155668/400000 [00:22<00:33, 7300.93it/s] 39%|███▉      | 156400/400000 [00:22<00:33, 7278.39it/s] 39%|███▉      | 157129/400000 [00:22<00:33, 7200.45it/s] 39%|███▉      | 157891/400000 [00:22<00:33, 7320.96it/s] 40%|███▉      | 158626/400000 [00:22<00:32, 7328.70it/s] 40%|███▉      | 159360/400000 [00:22<00:32, 7323.14it/s] 40%|████      | 160093/400000 [00:22<00:32, 7309.08it/s] 40%|████      | 160825/400000 [00:22<00:33, 7246.70it/s] 40%|████      | 161557/400000 [00:22<00:32, 7265.84it/s] 41%|████      | 162320/400000 [00:22<00:32, 7369.67it/s] 41%|████      | 163066/400000 [00:23<00:32, 7393.61it/s] 41%|████      | 163806/400000 [00:23<00:32, 7307.22it/s] 41%|████      | 164538/400000 [00:23<00:32, 7175.46it/s] 41%|████▏     | 165261/400000 [00:23<00:32, 7191.47it/s] 41%|████▏     | 165981/400000 [00:23<00:32, 7165.47it/s] 42%|████▏     | 166748/400000 [00:23<00:31, 7307.07it/s] 42%|████▏     | 167480/400000 [00:23<00:32, 7250.41it/s] 42%|████▏     | 168206/400000 [00:23<00:34, 6766.64it/s] 42%|████▏     | 168912/400000 [00:23<00:33, 6850.41it/s] 42%|████▏     | 169641/400000 [00:24<00:33, 6974.24it/s] 43%|████▎     | 170357/400000 [00:24<00:32, 7026.75it/s] 43%|████▎     | 171096/400000 [00:24<00:32, 7131.46it/s] 43%|████▎     | 171812/400000 [00:24<00:32, 7097.76it/s] 43%|████▎     | 172524/400000 [00:24<00:33, 6869.24it/s] 43%|████▎     | 173214/400000 [00:24<00:36, 6236.57it/s] 43%|████▎     | 173954/400000 [00:24<00:34, 6545.10it/s] 44%|████▎     | 174699/400000 [00:24<00:33, 6791.63it/s] 44%|████▍     | 175412/400000 [00:24<00:32, 6889.18it/s] 44%|████▍     | 176135/400000 [00:24<00:32, 6987.88it/s] 44%|████▍     | 176841/400000 [00:25<00:32, 6913.41it/s] 44%|████▍     | 177558/400000 [00:25<00:31, 6987.09it/s] 45%|████▍     | 178277/400000 [00:25<00:31, 7045.27it/s] 45%|████▍     | 178985/400000 [00:25<00:31, 7038.80it/s] 45%|████▍     | 179718/400000 [00:25<00:30, 7123.10it/s] 45%|████▌     | 180432/400000 [00:25<00:31, 6981.43it/s] 45%|████▌     | 181186/400000 [00:25<00:30, 7139.02it/s] 45%|████▌     | 181911/400000 [00:25<00:30, 7171.42it/s] 46%|████▌     | 182659/400000 [00:25<00:29, 7259.03it/s] 46%|████▌     | 183403/400000 [00:25<00:29, 7310.67it/s] 46%|████▌     | 184136/400000 [00:26<00:30, 7193.77it/s] 46%|████▌     | 184857/400000 [00:26<00:29, 7172.36it/s] 46%|████▋     | 185601/400000 [00:26<00:29, 7249.57it/s] 47%|████▋     | 186351/400000 [00:26<00:29, 7321.33it/s] 47%|████▋     | 187114/400000 [00:26<00:28, 7408.96it/s] 47%|████▋     | 187856/400000 [00:26<00:29, 7126.07it/s] 47%|████▋     | 188588/400000 [00:26<00:29, 7181.35it/s] 47%|████▋     | 189309/400000 [00:26<00:29, 7151.81it/s] 48%|████▊     | 190042/400000 [00:26<00:29, 7202.55it/s] 48%|████▊     | 190774/400000 [00:26<00:28, 7235.10it/s] 48%|████▊     | 191499/400000 [00:27<00:29, 7121.54it/s] 48%|████▊     | 192217/400000 [00:27<00:29, 7137.21it/s] 48%|████▊     | 192949/400000 [00:27<00:28, 7190.19it/s] 48%|████▊     | 193713/400000 [00:27<00:28, 7318.50it/s] 49%|████▊     | 194448/400000 [00:27<00:28, 7326.69it/s] 49%|████▉     | 195200/400000 [00:27<00:27, 7382.71it/s] 49%|████▉     | 195946/400000 [00:27<00:27, 7404.28it/s] 49%|████▉     | 196687/400000 [00:27<00:27, 7290.50it/s] 49%|████▉     | 197417/400000 [00:27<00:28, 7045.35it/s] 50%|████▉     | 198124/400000 [00:28<00:29, 6886.07it/s] 50%|████▉     | 198815/400000 [00:28<00:29, 6788.34it/s] 50%|████▉     | 199496/400000 [00:28<00:30, 6587.24it/s] 50%|█████     | 200158/400000 [00:28<00:30, 6522.56it/s] 50%|█████     | 200819/400000 [00:28<00:30, 6546.13it/s] 50%|█████     | 201547/400000 [00:28<00:29, 6748.93it/s] 51%|█████     | 202225/400000 [00:28<00:29, 6672.25it/s] 51%|█████     | 202923/400000 [00:28<00:29, 6756.89it/s] 51%|█████     | 203670/400000 [00:28<00:28, 6954.15it/s] 51%|█████     | 204383/400000 [00:28<00:27, 7003.50it/s] 51%|█████▏    | 205138/400000 [00:29<00:27, 7157.24it/s] 51%|█████▏    | 205856/400000 [00:29<00:27, 7006.25it/s] 52%|█████▏    | 206573/400000 [00:29<00:27, 7052.57it/s] 52%|█████▏    | 207280/400000 [00:29<00:27, 7013.44it/s] 52%|█████▏    | 207983/400000 [00:29<00:28, 6779.25it/s] 52%|█████▏    | 208737/400000 [00:29<00:27, 6990.61it/s] 52%|█████▏    | 209440/400000 [00:29<00:27, 6990.83it/s] 53%|█████▎    | 210142/400000 [00:29<00:27, 6840.87it/s] 53%|█████▎    | 210829/400000 [00:29<00:27, 6836.44it/s] 53%|█████▎    | 211552/400000 [00:29<00:27, 6947.89it/s] 53%|█████▎    | 212299/400000 [00:30<00:26, 7095.05it/s] 53%|█████▎    | 213047/400000 [00:30<00:25, 7204.13it/s] 53%|█████▎    | 213779/400000 [00:30<00:25, 7236.56it/s] 54%|█████▎    | 214504/400000 [00:30<00:25, 7230.87it/s] 54%|█████▍    | 215252/400000 [00:30<00:25, 7303.10it/s] 54%|█████▍    | 216001/400000 [00:30<00:25, 7357.92it/s] 54%|█████▍    | 216746/400000 [00:30<00:24, 7383.17it/s] 54%|█████▍    | 217501/400000 [00:30<00:24, 7430.23it/s] 55%|█████▍    | 218245/400000 [00:30<00:24, 7391.78it/s] 55%|█████▍    | 218985/400000 [00:30<00:25, 7225.56it/s] 55%|█████▍    | 219731/400000 [00:31<00:24, 7293.40it/s] 55%|█████▌    | 220462/400000 [00:31<00:24, 7254.40it/s] 55%|█████▌    | 221189/400000 [00:31<00:24, 7183.02it/s] 55%|█████▌    | 221916/400000 [00:31<00:24, 7208.12it/s] 56%|█████▌    | 222638/400000 [00:31<00:24, 7110.15it/s] 56%|█████▌    | 223387/400000 [00:31<00:24, 7216.93it/s] 56%|█████▌    | 224136/400000 [00:31<00:24, 7295.09it/s] 56%|█████▌    | 224868/400000 [00:31<00:23, 7302.52it/s] 56%|█████▋    | 225601/400000 [00:31<00:23, 7285.88it/s] 57%|█████▋    | 226330/400000 [00:32<00:24, 7164.55it/s] 57%|█████▋    | 227048/400000 [00:32<00:24, 6991.14it/s] 57%|█████▋    | 227750/400000 [00:32<00:24, 6998.09it/s] 57%|█████▋    | 228490/400000 [00:32<00:24, 7112.25it/s] 57%|█████▋    | 229205/400000 [00:32<00:23, 7123.22it/s] 57%|█████▋    | 229919/400000 [00:32<00:24, 6883.52it/s] 58%|█████▊    | 230610/400000 [00:32<00:24, 6836.53it/s] 58%|█████▊    | 231340/400000 [00:32<00:24, 6967.69it/s] 58%|█████▊    | 232091/400000 [00:32<00:23, 7119.63it/s] 58%|█████▊    | 232823/400000 [00:32<00:23, 7177.87it/s] 58%|█████▊    | 233543/400000 [00:33<00:23, 6956.19it/s] 59%|█████▊    | 234257/400000 [00:33<00:23, 7008.50it/s] 59%|█████▊    | 234968/400000 [00:33<00:23, 7037.84it/s] 59%|█████▉    | 235674/400000 [00:33<00:23, 6879.14it/s] 59%|█████▉    | 236407/400000 [00:33<00:23, 7006.84it/s] 59%|█████▉    | 237141/400000 [00:33<00:22, 7103.29it/s] 59%|█████▉    | 237909/400000 [00:33<00:22, 7258.24it/s] 60%|█████▉    | 238637/400000 [00:33<00:22, 7189.56it/s] 60%|█████▉    | 239358/400000 [00:33<00:22, 7131.14it/s] 60%|██████    | 240078/400000 [00:33<00:22, 7150.27it/s] 60%|██████    | 240825/400000 [00:34<00:21, 7240.88it/s] 60%|██████    | 241550/400000 [00:34<00:21, 7221.48it/s] 61%|██████    | 242273/400000 [00:34<00:22, 6910.64it/s] 61%|██████    | 242968/400000 [00:34<00:22, 6906.29it/s] 61%|██████    | 243709/400000 [00:34<00:22, 7049.86it/s] 61%|██████    | 244445/400000 [00:34<00:21, 7138.32it/s] 61%|██████▏   | 245191/400000 [00:34<00:21, 7230.92it/s] 61%|██████▏   | 245926/400000 [00:34<00:21, 7264.58it/s] 62%|██████▏   | 246654/400000 [00:34<00:21, 7010.40it/s] 62%|██████▏   | 247381/400000 [00:34<00:21, 7084.52it/s] 62%|██████▏   | 248092/400000 [00:35<00:22, 6900.97it/s] 62%|██████▏   | 248785/400000 [00:35<00:22, 6868.18it/s] 62%|██████▏   | 249505/400000 [00:35<00:21, 6962.41it/s] 63%|██████▎   | 250238/400000 [00:35<00:21, 7068.26it/s] 63%|██████▎   | 250973/400000 [00:35<00:20, 7148.30it/s] 63%|██████▎   | 251745/400000 [00:35<00:20, 7310.08it/s] 63%|██████▎   | 252498/400000 [00:35<00:20, 7372.44it/s] 63%|██████▎   | 253252/400000 [00:35<00:19, 7419.78it/s] 63%|██████▎   | 253996/400000 [00:35<00:20, 7168.83it/s] 64%|██████▎   | 254716/400000 [00:36<00:20, 6998.28it/s] 64%|██████▍   | 255459/400000 [00:36<00:20, 7121.47it/s] 64%|██████▍   | 256200/400000 [00:36<00:19, 7203.55it/s] 64%|██████▍   | 256935/400000 [00:36<00:19, 7244.69it/s] 64%|██████▍   | 257661/400000 [00:36<00:19, 7236.00it/s] 65%|██████▍   | 258386/400000 [00:36<00:19, 7111.77it/s] 65%|██████▍   | 259114/400000 [00:36<00:19, 7160.82it/s] 65%|██████▍   | 259856/400000 [00:36<00:19, 7233.92it/s] 65%|██████▌   | 260581/400000 [00:36<00:19, 7214.46it/s] 65%|██████▌   | 261304/400000 [00:36<00:19, 7156.61it/s] 66%|██████▌   | 262021/400000 [00:37<00:19, 7137.47it/s] 66%|██████▌   | 262759/400000 [00:37<00:19, 7207.95it/s] 66%|██████▌   | 263501/400000 [00:37<00:18, 7269.84it/s] 66%|██████▌   | 264240/400000 [00:37<00:18, 7303.69it/s] 66%|██████▌   | 264971/400000 [00:37<00:18, 7130.52it/s] 66%|██████▋   | 265686/400000 [00:37<00:19, 6874.43it/s] 67%|██████▋   | 266377/400000 [00:37<00:20, 6639.73it/s] 67%|██████▋   | 267069/400000 [00:37<00:19, 6720.55it/s] 67%|██████▋   | 267783/400000 [00:37<00:19, 6840.82it/s] 67%|██████▋   | 268470/400000 [00:37<00:20, 6533.30it/s] 67%|██████▋   | 269128/400000 [00:38<00:20, 6479.82it/s] 67%|██████▋   | 269838/400000 [00:38<00:19, 6653.87it/s] 68%|██████▊   | 270567/400000 [00:38<00:18, 6830.99it/s] 68%|██████▊   | 271254/400000 [00:38<00:19, 6679.60it/s] 68%|██████▊   | 271938/400000 [00:38<00:19, 6725.21it/s] 68%|██████▊   | 272633/400000 [00:38<00:18, 6788.63it/s] 68%|██████▊   | 273352/400000 [00:38<00:18, 6903.42it/s] 69%|██████▊   | 274074/400000 [00:38<00:18, 6992.44it/s] 69%|██████▊   | 274819/400000 [00:38<00:17, 7122.74it/s] 69%|██████▉   | 275535/400000 [00:38<00:17, 7131.09it/s] 69%|██████▉   | 276250/400000 [00:39<00:17, 7044.01it/s] 69%|██████▉   | 276956/400000 [00:39<00:17, 6986.35it/s] 69%|██████▉   | 277682/400000 [00:39<00:17, 7065.55it/s] 70%|██████▉   | 278390/400000 [00:39<00:17, 7056.41it/s] 70%|██████▉   | 279102/400000 [00:39<00:17, 7073.21it/s] 70%|██████▉   | 279827/400000 [00:39<00:16, 7124.92it/s] 70%|███████   | 280557/400000 [00:39<00:16, 7175.25it/s] 70%|███████   | 281275/400000 [00:39<00:16, 7125.61it/s] 71%|███████   | 282027/400000 [00:39<00:16, 7238.77it/s] 71%|███████   | 282769/400000 [00:40<00:16, 7290.11it/s] 71%|███████   | 283506/400000 [00:40<00:15, 7306.23it/s] 71%|███████   | 284253/400000 [00:40<00:15, 7353.96it/s] 71%|███████   | 284989/400000 [00:40<00:16, 6882.75it/s] 71%|███████▏  | 285684/400000 [00:40<00:16, 6899.82it/s] 72%|███████▏  | 286432/400000 [00:40<00:16, 7062.06it/s] 72%|███████▏  | 287143/400000 [00:40<00:16, 7036.69it/s] 72%|███████▏  | 287850/400000 [00:40<00:15, 7044.38it/s] 72%|███████▏  | 288557/400000 [00:40<00:15, 6976.03it/s] 72%|███████▏  | 289289/400000 [00:40<00:15, 7075.08it/s] 72%|███████▏  | 289998/400000 [00:41<00:15, 6949.57it/s] 73%|███████▎  | 290695/400000 [00:41<00:16, 6772.02it/s] 73%|███████▎  | 291399/400000 [00:41<00:15, 6848.34it/s] 73%|███████▎  | 292086/400000 [00:41<00:15, 6830.39it/s] 73%|███████▎  | 292802/400000 [00:41<00:15, 6925.37it/s] 73%|███████▎  | 293496/400000 [00:41<00:15, 6870.73it/s] 74%|███████▎  | 294190/400000 [00:41<00:15, 6885.85it/s] 74%|███████▎  | 294897/400000 [00:41<00:15, 6937.34it/s] 74%|███████▍  | 295620/400000 [00:41<00:14, 7022.37it/s] 74%|███████▍  | 296323/400000 [00:41<00:14, 6944.86it/s] 74%|███████▍  | 297026/400000 [00:42<00:14, 6966.56it/s] 74%|███████▍  | 297768/400000 [00:42<00:14, 7094.11it/s] 75%|███████▍  | 298520/400000 [00:42<00:14, 7216.33it/s] 75%|███████▍  | 299243/400000 [00:42<00:15, 6694.56it/s] 75%|███████▍  | 299923/400000 [00:42<00:14, 6723.68it/s] 75%|███████▌  | 300622/400000 [00:42<00:14, 6801.38it/s] 75%|███████▌  | 301307/400000 [00:42<00:14, 6726.22it/s] 75%|███████▌  | 301994/400000 [00:42<00:14, 6768.71it/s] 76%|███████▌  | 302688/400000 [00:42<00:14, 6819.18it/s] 76%|███████▌  | 303409/400000 [00:42<00:13, 6930.23it/s] 76%|███████▌  | 304136/400000 [00:43<00:13, 7026.30it/s] 76%|███████▌  | 304850/400000 [00:43<00:13, 7059.18it/s] 76%|███████▋  | 305569/400000 [00:43<00:13, 7096.72it/s] 77%|███████▋  | 306280/400000 [00:43<00:14, 6646.19it/s] 77%|███████▋  | 306951/400000 [00:43<00:14, 6449.35it/s] 77%|███████▋  | 307675/400000 [00:43<00:13, 6666.48it/s] 77%|███████▋  | 308427/400000 [00:43<00:13, 6899.20it/s] 77%|███████▋  | 309188/400000 [00:43<00:12, 7096.25it/s] 77%|███████▋  | 309904/400000 [00:43<00:12, 7049.60it/s] 78%|███████▊  | 310614/400000 [00:44<00:12, 7036.60it/s] 78%|███████▊  | 311359/400000 [00:44<00:12, 7155.07it/s] 78%|███████▊  | 312077/400000 [00:44<00:12, 7016.29it/s] 78%|███████▊  | 312840/400000 [00:44<00:12, 7189.14it/s] 78%|███████▊  | 313585/400000 [00:44<00:11, 7263.19it/s] 79%|███████▊  | 314339/400000 [00:44<00:11, 7343.28it/s] 79%|███████▉  | 315085/400000 [00:44<00:11, 7375.98it/s] 79%|███████▉  | 315824/400000 [00:44<00:11, 7130.76it/s] 79%|███████▉  | 316552/400000 [00:44<00:11, 7174.38it/s] 79%|███████▉  | 317272/400000 [00:44<00:11, 6913.34it/s] 80%|███████▉  | 318022/400000 [00:45<00:11, 7077.80it/s] 80%|███████▉  | 318745/400000 [00:45<00:11, 7121.43it/s] 80%|███████▉  | 319468/400000 [00:45<00:11, 7151.05it/s] 80%|████████  | 320234/400000 [00:45<00:10, 7294.21it/s] 80%|████████  | 320966/400000 [00:45<00:10, 7245.93it/s] 80%|████████  | 321692/400000 [00:45<00:11, 6831.39it/s] 81%|████████  | 322381/400000 [00:45<00:11, 6718.01it/s] 81%|████████  | 323120/400000 [00:45<00:11, 6903.79it/s] 81%|████████  | 323869/400000 [00:45<00:10, 7067.22it/s] 81%|████████  | 324580/400000 [00:46<00:10, 6928.96it/s] 81%|████████▏ | 325277/400000 [00:46<00:10, 6935.83it/s] 82%|████████▏ | 326010/400000 [00:46<00:10, 7048.47it/s] 82%|████████▏ | 326739/400000 [00:46<00:10, 7117.15it/s] 82%|████████▏ | 327500/400000 [00:46<00:09, 7257.75it/s] 82%|████████▏ | 328228/400000 [00:46<00:10, 7148.78it/s] 82%|████████▏ | 328951/400000 [00:46<00:09, 7171.84it/s] 82%|████████▏ | 329670/400000 [00:46<00:10, 6941.61it/s] 83%|████████▎ | 330367/400000 [00:46<00:10, 6836.56it/s] 83%|████████▎ | 331053/400000 [00:46<00:10, 6594.17it/s] 83%|████████▎ | 331716/400000 [00:47<00:10, 6461.64it/s] 83%|████████▎ | 332451/400000 [00:47<00:10, 6703.65it/s] 83%|████████▎ | 333146/400000 [00:47<00:09, 6775.25it/s] 83%|████████▎ | 333856/400000 [00:47<00:09, 6868.62it/s] 84%|████████▎ | 334615/400000 [00:47<00:09, 7069.64it/s] 84%|████████▍ | 335348/400000 [00:47<00:09, 7143.81it/s] 84%|████████▍ | 336065/400000 [00:47<00:08, 7126.98it/s] 84%|████████▍ | 336780/400000 [00:47<00:09, 6975.03it/s] 84%|████████▍ | 337551/400000 [00:47<00:08, 7180.01it/s] 85%|████████▍ | 338272/400000 [00:47<00:08, 7064.11it/s] 85%|████████▍ | 338981/400000 [00:48<00:08, 7024.28it/s] 85%|████████▍ | 339717/400000 [00:48<00:08, 7121.73it/s] 85%|████████▌ | 340431/400000 [00:48<00:08, 6951.90it/s] 85%|████████▌ | 341129/400000 [00:48<00:08, 6791.05it/s] 85%|████████▌ | 341875/400000 [00:48<00:08, 6977.50it/s] 86%|████████▌ | 342576/400000 [00:48<00:08, 6718.04it/s] 86%|████████▌ | 343252/400000 [00:48<00:08, 6622.43it/s] 86%|████████▌ | 343918/400000 [00:48<00:08, 6471.15it/s] 86%|████████▌ | 344632/400000 [00:48<00:08, 6642.46it/s] 86%|████████▋ | 345354/400000 [00:49<00:08, 6802.87it/s] 87%|████████▋ | 346038/400000 [00:49<00:08, 6729.87it/s] 87%|████████▋ | 346749/400000 [00:49<00:07, 6837.78it/s] 87%|████████▋ | 347473/400000 [00:49<00:07, 6951.21it/s] 87%|████████▋ | 348212/400000 [00:49<00:07, 7076.16it/s] 87%|████████▋ | 348927/400000 [00:49<00:07, 7096.11it/s] 87%|████████▋ | 349638/400000 [00:49<00:07, 6803.44it/s] 88%|████████▊ | 350380/400000 [00:49<00:07, 6976.11it/s] 88%|████████▊ | 351133/400000 [00:49<00:06, 7131.92it/s] 88%|████████▊ | 351854/400000 [00:49<00:06, 7154.87it/s] 88%|████████▊ | 352596/400000 [00:50<00:06, 7230.18it/s] 88%|████████▊ | 353321/400000 [00:50<00:06, 7117.85it/s] 89%|████████▊ | 354064/400000 [00:50<00:06, 7208.62it/s] 89%|████████▊ | 354817/400000 [00:50<00:06, 7301.63it/s] 89%|████████▉ | 355549/400000 [00:50<00:06, 6993.52it/s] 89%|████████▉ | 356253/400000 [00:50<00:06, 6923.87it/s] 89%|████████▉ | 356975/400000 [00:50<00:06, 7001.25it/s] 89%|████████▉ | 357678/400000 [00:50<00:06, 6970.25it/s] 90%|████████▉ | 358429/400000 [00:50<00:05, 7122.64it/s] 90%|████████▉ | 359181/400000 [00:50<00:05, 7229.64it/s] 90%|████████▉ | 359942/400000 [00:51<00:05, 7337.35it/s] 90%|█████████ | 360697/400000 [00:51<00:05, 7398.33it/s] 90%|█████████ | 361439/400000 [00:51<00:05, 7347.63it/s] 91%|█████████ | 362175/400000 [00:51<00:05, 7331.71it/s] 91%|█████████ | 362909/400000 [00:51<00:05, 7329.85it/s] 91%|█████████ | 363643/400000 [00:51<00:05, 7112.22it/s] 91%|█████████ | 364402/400000 [00:51<00:04, 7249.07it/s] 91%|█████████▏| 365129/400000 [00:51<00:04, 7044.21it/s] 91%|█████████▏| 365836/400000 [00:51<00:04, 7048.52it/s] 92%|█████████▏| 366576/400000 [00:51<00:04, 7149.29it/s] 92%|█████████▏| 367293/400000 [00:52<00:04, 7108.30it/s] 92%|█████████▏| 368048/400000 [00:52<00:04, 7234.74it/s] 92%|█████████▏| 368801/400000 [00:52<00:04, 7319.75it/s] 92%|█████████▏| 369540/400000 [00:52<00:04, 7337.64it/s] 93%|█████████▎| 370289/400000 [00:52<00:04, 7381.21it/s] 93%|█████████▎| 371053/400000 [00:52<00:03, 7456.08it/s] 93%|█████████▎| 371811/400000 [00:52<00:03, 7492.33it/s] 93%|█████████▎| 372561/400000 [00:52<00:03, 7369.36it/s] 93%|█████████▎| 373299/400000 [00:52<00:03, 6988.43it/s] 94%|█████████▎| 374003/400000 [00:53<00:03, 6875.69it/s] 94%|█████████▎| 374741/400000 [00:53<00:03, 7019.41it/s] 94%|█████████▍| 375492/400000 [00:53<00:03, 7159.23it/s] 94%|█████████▍| 376232/400000 [00:53<00:03, 7228.65it/s] 94%|█████████▍| 376983/400000 [00:53<00:03, 7309.48it/s] 94%|█████████▍| 377716/400000 [00:53<00:03, 7197.65it/s] 95%|█████████▍| 378438/400000 [00:53<00:02, 7192.14it/s] 95%|█████████▍| 379163/400000 [00:53<00:02, 7208.17it/s] 95%|█████████▍| 379910/400000 [00:53<00:02, 7282.06it/s] 95%|█████████▌| 380639/400000 [00:53<00:02, 7188.84it/s] 95%|█████████▌| 381393/400000 [00:54<00:02, 7288.26it/s] 96%|█████████▌| 382148/400000 [00:54<00:02, 7363.63it/s] 96%|█████████▌| 382907/400000 [00:54<00:02, 7429.00it/s] 96%|█████████▌| 383654/400000 [00:54<00:02, 7441.05it/s] 96%|█████████▌| 384408/400000 [00:54<00:02, 7469.87it/s] 96%|█████████▋| 385156/400000 [00:54<00:02, 7405.88it/s] 96%|█████████▋| 385916/400000 [00:54<00:01, 7462.91it/s] 97%|█████████▋| 386669/400000 [00:54<00:01, 7482.63it/s] 97%|█████████▋| 387418/400000 [00:54<00:01, 7459.72it/s] 97%|█████████▋| 388169/400000 [00:54<00:01, 7473.55it/s] 97%|█████████▋| 388917/400000 [00:55<00:01, 7470.90it/s] 97%|█████████▋| 389665/400000 [00:55<00:01, 7355.26it/s] 98%|█████████▊| 390402/400000 [00:55<00:01, 7274.78it/s] 98%|█████████▊| 391131/400000 [00:55<00:01, 7048.88it/s] 98%|█████████▊| 391852/400000 [00:55<00:01, 7095.94it/s] 98%|█████████▊| 392611/400000 [00:55<00:01, 7237.26it/s] 98%|█████████▊| 393370/400000 [00:55<00:00, 7337.12it/s] 99%|█████████▊| 394114/400000 [00:55<00:00, 7366.08it/s] 99%|█████████▊| 394852/400000 [00:55<00:00, 7221.11it/s] 99%|█████████▉| 395603/400000 [00:55<00:00, 7304.31it/s] 99%|█████████▉| 396359/400000 [00:56<00:00, 7378.50it/s] 99%|█████████▉| 397127/400000 [00:56<00:00, 7464.43it/s] 99%|█████████▉| 397875/400000 [00:56<00:00, 7419.29it/s]100%|█████████▉| 398618/400000 [00:56<00:00, 7009.94it/s]100%|█████████▉| 399371/400000 [00:56<00:00, 7155.84it/s]100%|█████████▉| 399999/400000 [00:56<00:00, 7072.04it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f873c0116a0>, <torchtext.data.dataset.TabularDataset object at 0x7f873c0117f0>, <torchtext.vocab.Vocab object at 0x7f873c011710>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.59 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.59 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.59 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.51 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.73 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  3.73 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.32 file/s]2020-06-21 00:19:15.034167: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-06-21 00:19:15.038635: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-06-21 00:19:15.038825: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5562a0473270 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-21 00:19:15.038844: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 147406.59it/s] 81%|████████  | 8003584/9912422 [00:00<00:09, 210414.41it/s]9920512it [00:00, 43018943.76it/s]                           
0it [00:00, ?it/s]32768it [00:00, 579253.73it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 477552.43it/s]1654784it [00:00, 11974884.75it/s]                         
0it [00:00, ?it/s]8192it [00:00, 189322.37it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

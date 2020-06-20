
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': 'bd7dfc233939710e05244f8e6a394a7ce12a3485', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/bd7dfc233939710e05244f8e6a394a7ce12a3485

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/bd7dfc233939710e05244f8e6a394a7ce12a3485

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f37356aaea0> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f37356aaea0> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f37a0c61048> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f37a0c61048> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f37ba98fe18> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f37ba98fe18> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f374e4e08c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f374e4e08c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f374e4e08c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▍      | 3383296/9912422 [00:00<00:00, 32213814.71it/s]9920512it [00:00, 34715116.82it/s]                             
0it [00:00, ?it/s]32768it [00:00, 609579.99it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 472681.07it/s]1654784it [00:00, 11896152.03it/s]                         
0it [00:00, ?it/s]8192it [00:00, 180414.38it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f374ba4f550>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f374d4a07f0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f374e4e0510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f374e4e0510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f374e4e0510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.76 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:57,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:57,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:56,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.76 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:39,  3.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:39,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:26,  5.41 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:26,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.41 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:17,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 10.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:11, 10.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 14.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 14.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 19.13 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.13 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.05 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 31.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 39.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 39.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 47.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 47.98 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 55.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 55.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 62.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 69.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 69.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 73.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 73.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 76.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 79.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 79.17 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.18s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 79.17 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 38.19 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.24s/ url]
0 examples [00:00, ? examples/s]2020-06-20 06:08:57.911970: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-20 06:08:57.926090: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-20 06:08:57.926661: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56518818ae50 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-20 06:08:57.926682: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  2.97 examples/s]110 examples [00:00,  4.24 examples/s]221 examples [00:00,  6.05 examples/s]329 examples [00:00,  8.62 examples/s]440 examples [00:00, 12.28 examples/s]551 examples [00:00, 17.46 examples/s]659 examples [00:00, 24.77 examples/s]764 examples [00:01, 35.03 examples/s]873 examples [00:01, 49.36 examples/s]975 examples [00:01, 68.93 examples/s]1084 examples [00:01, 95.86 examples/s]1194 examples [00:01, 131.97 examples/s]1305 examples [00:01, 179.35 examples/s]1417 examples [00:01, 239.69 examples/s]1528 examples [00:01, 313.18 examples/s]1639 examples [00:01, 398.81 examples/s]1751 examples [00:01, 493.76 examples/s]1862 examples [00:02, 592.18 examples/s]1974 examples [00:02, 688.86 examples/s]2085 examples [00:02, 777.02 examples/s]2196 examples [00:02, 850.11 examples/s]2306 examples [00:02, 907.52 examples/s]2416 examples [00:02, 951.35 examples/s]2525 examples [00:02, 987.50 examples/s]2636 examples [00:02, 1020.47 examples/s]2746 examples [00:02, 1006.66 examples/s]2857 examples [00:02, 1033.50 examples/s]2968 examples [00:03, 1053.18 examples/s]3079 examples [00:03, 1067.27 examples/s]3190 examples [00:03, 1079.52 examples/s]3300 examples [00:03, 1078.26 examples/s]3411 examples [00:03, 1084.82 examples/s]3523 examples [00:03, 1093.52 examples/s]3633 examples [00:03, 1094.35 examples/s]3744 examples [00:03, 1097.17 examples/s]3854 examples [00:03, 1091.38 examples/s]3964 examples [00:03, 1093.04 examples/s]4074 examples [00:04, 1063.83 examples/s]4188 examples [00:04, 1083.91 examples/s]4297 examples [00:04, 1055.43 examples/s]4405 examples [00:04, 1061.58 examples/s]4512 examples [00:04, 1060.79 examples/s]4619 examples [00:04, 1040.16 examples/s]4731 examples [00:04, 1060.46 examples/s]4842 examples [00:04, 1074.75 examples/s]4951 examples [00:04, 1079.03 examples/s]5062 examples [00:05, 1087.03 examples/s]5174 examples [00:05, 1095.57 examples/s]5286 examples [00:05, 1101.72 examples/s]5397 examples [00:05, 1088.38 examples/s]5506 examples [00:05, 1078.57 examples/s]5614 examples [00:05, 999.29 examples/s] 5716 examples [00:05, 986.26 examples/s]5823 examples [00:05, 1008.26 examples/s]5928 examples [00:05, 1020.34 examples/s]6035 examples [00:05, 1032.41 examples/s]6139 examples [00:06, 1034.09 examples/s]6250 examples [00:06, 1054.59 examples/s]6356 examples [00:06, 1053.29 examples/s]6462 examples [00:06, 1039.39 examples/s]6567 examples [00:06, 1017.23 examples/s]6669 examples [00:06, 1004.40 examples/s]6774 examples [00:06, 1017.63 examples/s]6883 examples [00:06, 1037.00 examples/s]6988 examples [00:06, 1040.21 examples/s]7097 examples [00:06, 1052.27 examples/s]7203 examples [00:07, 1051.15 examples/s]7310 examples [00:07, 1056.33 examples/s]7418 examples [00:07, 1058.55 examples/s]7524 examples [00:07, 1030.48 examples/s]7631 examples [00:07, 1039.82 examples/s]7740 examples [00:07, 1052.92 examples/s]7850 examples [00:07, 1065.22 examples/s]7959 examples [00:07, 1070.59 examples/s]8067 examples [00:07, 1067.72 examples/s]8174 examples [00:07, 1066.43 examples/s]8283 examples [00:08, 1071.29 examples/s]8391 examples [00:08, 1073.07 examples/s]8499 examples [00:08, 1007.37 examples/s]8605 examples [00:08, 1022.38 examples/s]8712 examples [00:08, 1034.66 examples/s]8819 examples [00:08, 1043.69 examples/s]8929 examples [00:08, 1058.68 examples/s]9036 examples [00:08, 1051.27 examples/s]9143 examples [00:08, 1054.24 examples/s]9252 examples [00:09, 1064.38 examples/s]9362 examples [00:09, 1072.13 examples/s]9470 examples [00:09, 1069.28 examples/s]9580 examples [00:09, 1075.77 examples/s]9690 examples [00:09, 1081.23 examples/s]9800 examples [00:09, 1081.99 examples/s]9909 examples [00:09, 1084.06 examples/s]10018 examples [00:09, 1023.21 examples/s]10128 examples [00:09, 1044.53 examples/s]10234 examples [00:09, 1037.40 examples/s]10342 examples [00:10, 1047.26 examples/s]10450 examples [00:10, 1054.59 examples/s]10556 examples [00:10, 1046.89 examples/s]10666 examples [00:10, 1059.95 examples/s]10773 examples [00:10, 1034.12 examples/s]10877 examples [00:10, 1028.69 examples/s]10983 examples [00:10, 1037.24 examples/s]11091 examples [00:10, 1048.66 examples/s]11198 examples [00:10, 1054.81 examples/s]11305 examples [00:10, 1058.82 examples/s]11411 examples [00:11, 1035.06 examples/s]11518 examples [00:11, 1043.37 examples/s]11624 examples [00:11, 1045.58 examples/s]11731 examples [00:11, 1051.82 examples/s]11838 examples [00:11, 1054.83 examples/s]11945 examples [00:11, 1056.61 examples/s]12051 examples [00:11, 1053.64 examples/s]12159 examples [00:11, 1060.88 examples/s]12266 examples [00:11, 1056.53 examples/s]12373 examples [00:11, 1059.38 examples/s]12481 examples [00:12, 1065.01 examples/s]12591 examples [00:12, 1074.93 examples/s]12703 examples [00:12, 1087.08 examples/s]12812 examples [00:12, 1066.33 examples/s]12919 examples [00:12, 1034.46 examples/s]13023 examples [00:12, 946.93 examples/s] 13120 examples [00:12, 908.60 examples/s]13215 examples [00:12, 919.62 examples/s]13309 examples [00:12, 913.91 examples/s]13402 examples [00:13, 911.37 examples/s]13499 examples [00:13, 926.13 examples/s]13593 examples [00:13, 901.00 examples/s]13703 examples [00:13, 950.58 examples/s]13811 examples [00:13, 984.31 examples/s]13911 examples [00:13, 977.74 examples/s]14024 examples [00:13, 1016.33 examples/s]14135 examples [00:13, 1040.86 examples/s]14248 examples [00:13, 1064.91 examples/s]14362 examples [00:13, 1085.52 examples/s]14473 examples [00:14, 1091.47 examples/s]14585 examples [00:14, 1099.44 examples/s]14697 examples [00:14, 1103.72 examples/s]14808 examples [00:14, 1105.49 examples/s]14919 examples [00:14, 1097.14 examples/s]15029 examples [00:14, 1092.75 examples/s]15140 examples [00:14, 1096.37 examples/s]15250 examples [00:14, 1083.46 examples/s]15359 examples [00:14, 1072.17 examples/s]15468 examples [00:14, 1075.73 examples/s]15579 examples [00:15, 1083.66 examples/s]15689 examples [00:15, 1085.88 examples/s]15800 examples [00:15, 1091.62 examples/s]15911 examples [00:15, 1096.08 examples/s]16021 examples [00:15, 1091.22 examples/s]16133 examples [00:15, 1097.11 examples/s]16243 examples [00:15, 1096.38 examples/s]16354 examples [00:15, 1100.04 examples/s]16465 examples [00:15, 1101.71 examples/s]16576 examples [00:16, 1097.72 examples/s]16686 examples [00:16, 1084.80 examples/s]16795 examples [00:16, 1084.60 examples/s]16904 examples [00:16, 1083.55 examples/s]17015 examples [00:16, 1090.67 examples/s]17125 examples [00:16, 1092.12 examples/s]17235 examples [00:16, 1018.25 examples/s]17346 examples [00:16, 1044.01 examples/s]17457 examples [00:16, 1061.87 examples/s]17568 examples [00:16, 1073.69 examples/s]17678 examples [00:17, 1080.67 examples/s]17788 examples [00:17, 1086.21 examples/s]17899 examples [00:17, 1092.77 examples/s]18011 examples [00:17, 1100.35 examples/s]18122 examples [00:17, 1101.61 examples/s]18233 examples [00:17, 1097.46 examples/s]18343 examples [00:17, 1097.38 examples/s]18454 examples [00:17, 1098.21 examples/s]18566 examples [00:17, 1101.94 examples/s]18677 examples [00:17, 1100.39 examples/s]18788 examples [00:18, 1097.76 examples/s]18899 examples [00:18, 1099.31 examples/s]19009 examples [00:18, 1095.90 examples/s]19119 examples [00:18, 1096.15 examples/s]19229 examples [00:18, 1095.15 examples/s]19339 examples [00:18, 1092.96 examples/s]19450 examples [00:18, 1095.48 examples/s]19562 examples [00:18, 1100.37 examples/s]19674 examples [00:18, 1104.47 examples/s]19786 examples [00:18, 1108.74 examples/s]19897 examples [00:19, 1103.37 examples/s]20008 examples [00:19, 1045.73 examples/s]20119 examples [00:19, 1063.84 examples/s]20231 examples [00:19, 1077.40 examples/s]20340 examples [00:19, 1071.02 examples/s]20448 examples [00:19, 1060.84 examples/s]20555 examples [00:19, 1025.82 examples/s]20666 examples [00:19, 1048.61 examples/s]20772 examples [00:19, 1032.28 examples/s]20876 examples [00:20, 1022.59 examples/s]20985 examples [00:20, 1039.78 examples/s]21096 examples [00:20, 1057.67 examples/s]21207 examples [00:20, 1071.41 examples/s]21317 examples [00:20, 1078.83 examples/s]21427 examples [00:20, 1083.37 examples/s]21537 examples [00:20, 1086.64 examples/s]21648 examples [00:20, 1092.30 examples/s]21758 examples [00:20, 1091.29 examples/s]21868 examples [00:20, 1090.72 examples/s]21979 examples [00:21, 1093.85 examples/s]22089 examples [00:21, 1068.03 examples/s]22199 examples [00:21, 1074.86 examples/s]22307 examples [00:21, 1074.50 examples/s]22419 examples [00:21, 1086.13 examples/s]22531 examples [00:21, 1094.08 examples/s]22642 examples [00:21, 1097.13 examples/s]22752 examples [00:21, 1084.85 examples/s]22862 examples [00:21, 1088.26 examples/s]22974 examples [00:21, 1096.44 examples/s]23084 examples [00:22, 1097.36 examples/s]23194 examples [00:22, 1095.41 examples/s]23307 examples [00:22, 1103.88 examples/s]23418 examples [00:22, 1105.07 examples/s]23529 examples [00:22, 1106.44 examples/s]23641 examples [00:22, 1107.90 examples/s]23752 examples [00:22, 1106.91 examples/s]23863 examples [00:22, 1064.81 examples/s]23974 examples [00:22, 1076.77 examples/s]24084 examples [00:22, 1083.13 examples/s]24193 examples [00:23, 1078.11 examples/s]24301 examples [00:23, 1073.86 examples/s]24412 examples [00:23, 1082.31 examples/s]24522 examples [00:23, 1085.97 examples/s]24631 examples [00:23, 1072.62 examples/s]24739 examples [00:23, 1064.31 examples/s]24851 examples [00:23, 1078.57 examples/s]24968 examples [00:23, 1102.53 examples/s]25083 examples [00:23, 1113.66 examples/s]25196 examples [00:23, 1118.30 examples/s]25308 examples [00:24, 1111.22 examples/s]25420 examples [00:24, 1113.20 examples/s]25532 examples [00:24, 1114.18 examples/s]25644 examples [00:24, 1103.41 examples/s]25755 examples [00:24, 1102.47 examples/s]25866 examples [00:24, 1098.97 examples/s]25976 examples [00:24, 1093.76 examples/s]26087 examples [00:24, 1098.30 examples/s]26197 examples [00:24, 1094.79 examples/s]26307 examples [00:24, 1095.24 examples/s]26417 examples [00:25, 1084.57 examples/s]26526 examples [00:25, 1037.93 examples/s]26635 examples [00:25, 1052.34 examples/s]26745 examples [00:25, 1063.96 examples/s]26852 examples [00:25, 1062.53 examples/s]26964 examples [00:25, 1076.61 examples/s]27074 examples [00:25, 1082.92 examples/s]27183 examples [00:25, 1064.67 examples/s]27294 examples [00:25, 1077.86 examples/s]27405 examples [00:25, 1084.65 examples/s]27518 examples [00:26, 1095.35 examples/s]27629 examples [00:26, 1098.70 examples/s]27739 examples [00:26, 1084.16 examples/s]27849 examples [00:26, 1087.47 examples/s]27960 examples [00:26, 1092.34 examples/s]28070 examples [00:26, 1083.05 examples/s]28179 examples [00:26, 1070.97 examples/s]28288 examples [00:26, 1074.84 examples/s]28396 examples [00:26, 1075.37 examples/s]28508 examples [00:27, 1085.85 examples/s]28619 examples [00:27, 1092.74 examples/s]28729 examples [00:27, 1091.73 examples/s]28840 examples [00:27, 1096.75 examples/s]28951 examples [00:27, 1097.89 examples/s]29061 examples [00:27, 1078.44 examples/s]29172 examples [00:27, 1086.46 examples/s]29282 examples [00:27, 1089.86 examples/s]29393 examples [00:27, 1095.78 examples/s]29503 examples [00:27, 1089.39 examples/s]29614 examples [00:28, 1095.32 examples/s]29726 examples [00:28, 1100.19 examples/s]29837 examples [00:28, 1093.28 examples/s]29947 examples [00:28, 1063.45 examples/s]30054 examples [00:28, 989.11 examples/s] 30162 examples [00:28, 1012.85 examples/s]30274 examples [00:28, 1041.02 examples/s]30379 examples [00:28, 1026.45 examples/s]30483 examples [00:28, 1010.84 examples/s]30594 examples [00:28, 1037.82 examples/s]30705 examples [00:29, 1055.90 examples/s]30815 examples [00:29, 1068.28 examples/s]30923 examples [00:29, 1070.32 examples/s]31034 examples [00:29, 1080.63 examples/s]31145 examples [00:29, 1087.81 examples/s]31257 examples [00:29, 1096.18 examples/s]31370 examples [00:29, 1105.23 examples/s]31481 examples [00:29, 1102.67 examples/s]31592 examples [00:29, 1104.26 examples/s]31703 examples [00:29, 1100.40 examples/s]31814 examples [00:30, 1099.49 examples/s]31925 examples [00:30, 1100.04 examples/s]32036 examples [00:30, 1088.48 examples/s]32148 examples [00:30, 1095.53 examples/s]32259 examples [00:30, 1099.35 examples/s]32369 examples [00:30, 1084.37 examples/s]32479 examples [00:30, 1086.34 examples/s]32588 examples [00:30, 1067.26 examples/s]32700 examples [00:30, 1080.53 examples/s]32810 examples [00:31, 1084.81 examples/s]32922 examples [00:31, 1092.76 examples/s]33032 examples [00:31, 1080.86 examples/s]33141 examples [00:31, 1083.55 examples/s]33250 examples [00:31, 1037.22 examples/s]33355 examples [00:31, 1023.73 examples/s]33466 examples [00:31, 1046.73 examples/s]33572 examples [00:31, 1048.16 examples/s]33678 examples [00:31, 1018.23 examples/s]33781 examples [00:31, 1020.88 examples/s]33891 examples [00:32, 1041.53 examples/s]34002 examples [00:32, 1060.22 examples/s]34112 examples [00:32, 1071.25 examples/s]34223 examples [00:32, 1082.38 examples/s]34335 examples [00:32, 1092.80 examples/s]34446 examples [00:32, 1095.00 examples/s]34557 examples [00:32, 1097.94 examples/s]34667 examples [00:32, 1092.97 examples/s]34777 examples [00:32, 1094.78 examples/s]34893 examples [00:32, 1112.35 examples/s]35005 examples [00:33, 1110.17 examples/s]35117 examples [00:33, 1110.32 examples/s]35229 examples [00:33, 1111.18 examples/s]35341 examples [00:33, 1106.12 examples/s]35452 examples [00:33, 1099.69 examples/s]35564 examples [00:33, 1103.64 examples/s]35676 examples [00:33, 1107.42 examples/s]35788 examples [00:33, 1110.49 examples/s]35900 examples [00:33, 1106.89 examples/s]36012 examples [00:33, 1107.94 examples/s]36124 examples [00:34, 1109.37 examples/s]36236 examples [00:34, 1110.20 examples/s]36348 examples [00:34, 1112.05 examples/s]36460 examples [00:34, 1107.76 examples/s]36571 examples [00:34, 1104.92 examples/s]36682 examples [00:34, 1105.58 examples/s]36793 examples [00:34, 1106.21 examples/s]36904 examples [00:34, 1101.80 examples/s]37016 examples [00:34, 1104.63 examples/s]37127 examples [00:34, 1067.32 examples/s]37237 examples [00:35, 1075.04 examples/s]37345 examples [00:35, 1049.00 examples/s]37451 examples [00:35, 1029.30 examples/s]37560 examples [00:35, 1044.20 examples/s]37670 examples [00:35, 1058.89 examples/s]37777 examples [00:35, 1041.63 examples/s]37884 examples [00:35, 1047.62 examples/s]37989 examples [00:35, 1031.62 examples/s]38098 examples [00:35, 1046.95 examples/s]38210 examples [00:36, 1063.02 examples/s]38321 examples [00:36, 1075.90 examples/s]38429 examples [00:36, 1070.49 examples/s]38537 examples [00:36, 1071.22 examples/s]38647 examples [00:36, 1078.54 examples/s]38758 examples [00:36, 1086.41 examples/s]38868 examples [00:36, 1089.41 examples/s]38979 examples [00:36, 1093.37 examples/s]39089 examples [00:36, 1094.55 examples/s]39199 examples [00:36, 1066.88 examples/s]39307 examples [00:37, 1069.95 examples/s]39416 examples [00:37, 1075.78 examples/s]39527 examples [00:37, 1083.61 examples/s]39637 examples [00:37, 1088.05 examples/s]39746 examples [00:37, 1083.32 examples/s]39855 examples [00:37, 1058.51 examples/s]39966 examples [00:37, 1072.10 examples/s]40074 examples [00:37, 1024.73 examples/s]40183 examples [00:37, 1041.73 examples/s]40289 examples [00:37, 1045.22 examples/s]40394 examples [00:38, 1037.32 examples/s]40505 examples [00:38, 1055.92 examples/s]40616 examples [00:38, 1071.38 examples/s]40726 examples [00:38, 1079.08 examples/s]40835 examples [00:38, 1064.90 examples/s]40945 examples [00:38, 1073.82 examples/s]41056 examples [00:38, 1083.69 examples/s]41169 examples [00:38, 1094.94 examples/s]41279 examples [00:38, 1090.97 examples/s]41389 examples [00:38, 1083.67 examples/s]41498 examples [00:39, 1082.09 examples/s]41608 examples [00:39, 1086.42 examples/s]41720 examples [00:39, 1093.92 examples/s]41830 examples [00:39, 1092.52 examples/s]41940 examples [00:39, 1077.46 examples/s]42050 examples [00:39, 1081.40 examples/s]42161 examples [00:39, 1087.05 examples/s]42271 examples [00:39, 1089.46 examples/s]42381 examples [00:39, 1090.34 examples/s]42491 examples [00:39, 1088.81 examples/s]42601 examples [00:40, 1090.46 examples/s]42711 examples [00:40, 1088.14 examples/s]42820 examples [00:40, 1048.05 examples/s]42930 examples [00:40, 1062.65 examples/s]43039 examples [00:40, 1068.44 examples/s]43151 examples [00:40, 1083.18 examples/s]43260 examples [00:40, 1084.81 examples/s]43369 examples [00:40, 1078.52 examples/s]43478 examples [00:40, 1080.77 examples/s]43587 examples [00:41, 1077.27 examples/s]43695 examples [00:41, 1047.49 examples/s]43800 examples [00:41, 1027.73 examples/s]43904 examples [00:41, 1017.62 examples/s]44014 examples [00:41, 1040.96 examples/s]44119 examples [00:41, 1042.24 examples/s]44229 examples [00:41, 1057.24 examples/s]44340 examples [00:41, 1072.29 examples/s]44450 examples [00:41, 1078.32 examples/s]44558 examples [00:41, 1041.23 examples/s]44668 examples [00:42, 1056.68 examples/s]44778 examples [00:42, 1068.95 examples/s]44889 examples [00:42, 1079.20 examples/s]44999 examples [00:42, 1082.68 examples/s]45109 examples [00:42, 1086.13 examples/s]45220 examples [00:42, 1091.60 examples/s]45332 examples [00:42, 1099.81 examples/s]45443 examples [00:42, 1099.35 examples/s]45553 examples [00:42, 1096.43 examples/s]45663 examples [00:42, 1095.70 examples/s]45775 examples [00:43, 1102.48 examples/s]45886 examples [00:43, 1103.24 examples/s]45997 examples [00:43, 1093.14 examples/s]46107 examples [00:43, 1070.48 examples/s]46217 examples [00:43, 1078.30 examples/s]46325 examples [00:43, 1078.04 examples/s]46433 examples [00:43, 1056.49 examples/s]46542 examples [00:43, 1064.52 examples/s]46652 examples [00:43, 1072.27 examples/s]46761 examples [00:43, 1077.35 examples/s]46869 examples [00:44, 1055.34 examples/s]46975 examples [00:44, 1055.57 examples/s]47081 examples [00:44, 1043.69 examples/s]47192 examples [00:44, 1062.12 examples/s]47300 examples [00:44, 1066.76 examples/s]47410 examples [00:44, 1075.96 examples/s]47521 examples [00:44, 1084.85 examples/s]47630 examples [00:44, 1084.58 examples/s]47741 examples [00:44, 1089.05 examples/s]47850 examples [00:44, 1086.39 examples/s]47961 examples [00:45, 1093.18 examples/s]48072 examples [00:45, 1097.14 examples/s]48182 examples [00:45, 1026.84 examples/s]48286 examples [00:45, 1014.93 examples/s]48393 examples [00:45, 1029.97 examples/s]48504 examples [00:45, 1051.45 examples/s]48616 examples [00:45, 1068.98 examples/s]48725 examples [00:45, 1074.49 examples/s]48837 examples [00:45, 1084.99 examples/s]48948 examples [00:46, 1089.31 examples/s]49058 examples [00:46, 1088.99 examples/s]49168 examples [00:46, 1054.19 examples/s]49274 examples [00:46, 1047.61 examples/s]49386 examples [00:46, 1066.93 examples/s]49498 examples [00:46, 1080.28 examples/s]49611 examples [00:46, 1092.29 examples/s]49723 examples [00:46, 1100.43 examples/s]49836 examples [00:46, 1109.05 examples/s]49948 examples [00:46, 1108.31 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 6933/50000 [00:00<00:00, 69328.41 examples/s] 41%|████      | 20270/50000 [00:00<00:00, 80995.76 examples/s] 67%|██████▋   | 33546/50000 [00:00<00:00, 91724.96 examples/s] 94%|█████████▍| 47048/50000 [00:00<00:00, 101487.03 examples/s]                                                                0 examples [00:00, ? examples/s]92 examples [00:00, 919.37 examples/s]202 examples [00:00, 966.38 examples/s]314 examples [00:00, 1006.21 examples/s]427 examples [00:00, 1038.18 examples/s]539 examples [00:00, 1059.84 examples/s]652 examples [00:00, 1079.15 examples/s]763 examples [00:00, 1085.67 examples/s]874 examples [00:00, 1091.96 examples/s]985 examples [00:00, 1095.27 examples/s]1096 examples [00:01, 1098.21 examples/s]1208 examples [00:01, 1104.23 examples/s]1318 examples [00:01, 1102.55 examples/s]1428 examples [00:01, 1096.25 examples/s]1541 examples [00:01, 1103.40 examples/s]1652 examples [00:01, 1104.43 examples/s]1763 examples [00:01, 1106.01 examples/s]1874 examples [00:01, 1093.25 examples/s]1984 examples [00:01, 1067.82 examples/s]2095 examples [00:01, 1078.04 examples/s]2206 examples [00:02, 1087.12 examples/s]2319 examples [00:02, 1099.52 examples/s]2432 examples [00:02, 1105.46 examples/s]2543 examples [00:02, 1087.36 examples/s]2652 examples [00:02, 1083.16 examples/s]2764 examples [00:02, 1093.70 examples/s]2877 examples [00:02, 1102.97 examples/s]2988 examples [00:02, 1104.21 examples/s]3101 examples [00:02, 1109.73 examples/s]3213 examples [00:02, 1083.85 examples/s]3324 examples [00:03, 1089.58 examples/s]3434 examples [00:03, 1089.65 examples/s]3544 examples [00:03, 1082.31 examples/s]3657 examples [00:03, 1093.40 examples/s]3771 examples [00:03, 1104.48 examples/s]3882 examples [00:03, 1100.26 examples/s]3995 examples [00:03, 1108.76 examples/s]4107 examples [00:03, 1110.10 examples/s]4219 examples [00:03, 1112.84 examples/s]4331 examples [00:03, 1100.54 examples/s]4442 examples [00:04, 1096.47 examples/s]4552 examples [00:04, 1089.34 examples/s]4661 examples [00:04, 1084.49 examples/s]4771 examples [00:04, 1087.89 examples/s]4880 examples [00:04, 1086.81 examples/s]4989 examples [00:04, 1073.55 examples/s]5097 examples [00:04, 1064.63 examples/s]5207 examples [00:04, 1072.72 examples/s]5319 examples [00:04, 1085.27 examples/s]5431 examples [00:04, 1093.16 examples/s]5543 examples [00:05, 1099.71 examples/s]5654 examples [00:05, 1096.51 examples/s]5764 examples [00:05, 1080.52 examples/s]5873 examples [00:05, 1069.65 examples/s]5984 examples [00:05, 1080.70 examples/s]6095 examples [00:05, 1087.38 examples/s]6209 examples [00:05, 1100.16 examples/s]6320 examples [00:05, 1075.66 examples/s]6428 examples [00:05, 1075.75 examples/s]6536 examples [00:05, 1062.35 examples/s]6644 examples [00:06, 1066.66 examples/s]6757 examples [00:06, 1082.48 examples/s]6866 examples [00:06, 1078.96 examples/s]6976 examples [00:06, 1084.53 examples/s]7087 examples [00:06, 1091.96 examples/s]7198 examples [00:06, 1096.40 examples/s]7309 examples [00:06, 1099.37 examples/s]7419 examples [00:06, 1097.46 examples/s]7529 examples [00:06, 1084.71 examples/s]7640 examples [00:07, 1089.83 examples/s]7750 examples [00:07, 1089.44 examples/s]7862 examples [00:07, 1098.34 examples/s]7972 examples [00:07, 1093.34 examples/s]8082 examples [00:07, 1092.81 examples/s]8192 examples [00:07, 1062.50 examples/s]8303 examples [00:07, 1074.77 examples/s]8414 examples [00:07, 1083.63 examples/s]8523 examples [00:07, 1082.63 examples/s]8632 examples [00:07, 1059.87 examples/s]8741 examples [00:08, 1067.17 examples/s]8848 examples [00:08, 1047.68 examples/s]8961 examples [00:08, 1067.58 examples/s]9072 examples [00:08, 1078.97 examples/s]9181 examples [00:08, 1064.55 examples/s]9291 examples [00:08, 1072.66 examples/s]9401 examples [00:08, 1078.53 examples/s]9514 examples [00:08, 1092.80 examples/s]9625 examples [00:08, 1095.89 examples/s]9735 examples [00:08, 1049.89 examples/s]9841 examples [00:09, 1033.49 examples/s]9951 examples [00:09, 1051.74 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteF2QFGF/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteF2QFGF/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f374e4e08c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f374e4e08c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f374e4e08c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f36e46713c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f36e47340b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f374e4e08c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f374e4e08c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f374e4e08c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f374d4a03c8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f374d4a0240>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f36c601f400> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f36c601f400> 

  function with postional parmater data_info <function split_train_valid at 0x7f36c601f400> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f36c601f510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f36c601f510> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f36c601f510> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.3.0
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.0/en_core_web_sm-2.3.0.tar.gz (12.0 MB)
Requirement already satisfied: spacy<2.4.0,>=2.3.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.3.0) (2.3.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (4.46.1)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.0.3)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.24.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.1.3)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.18.5)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.4.1)
Requirement already satisfied: thinc==7.4.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (7.4.1)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (0.6.0)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (45.2.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.0.0)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (2.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.25.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy<2.4.0,>=2.3.0->en_core_web_sm==2.3.0) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.3.0-py3-none-any.whl size=12048606 sha256=5d3377528fa903d4b2e5a83908ccb91e7b27f631a889c71fa9643a5a26a9e79f
  Stored in directory: /tmp/pip-ephem-wheel-cache-lp_dn89y/wheels/4a/db/07/94eee4f3a60150464a04160bd0dfe9c8752ab981fe92f16aea
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<20:59:33, 11.4kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:55:52, 16.0kB/s].vector_cache/glove.6B.zip:   0%|          | 213k/862M [00:00<10:29:59, 22.8kB/s] .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<7:21:37, 32.5kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.59M/862M [00:01<5:08:21, 46.4kB/s].vector_cache/glove.6B.zip:   1%|          | 7.13M/862M [00:01<3:35:04, 66.3kB/s].vector_cache/glove.6B.zip:   1%|▏         | 11.5M/862M [00:01<2:29:53, 94.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 15.6M/862M [00:01<1:44:31, 135kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 20.1M/862M [00:01<1:12:52, 193kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.1M/862M [00:01<50:52, 275kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 28.7M/862M [00:01<35:30, 391kB/s].vector_cache/glove.6B.zip:   4%|▍         | 32.7M/862M [00:01<24:50, 556kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.2M/862M [00:02<17:23, 791kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.3M/862M [00:02<12:13, 1.12MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.6M/862M [00:02<08:42, 1.57MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.4M/862M [00:02<06:12, 2.19MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:02<04:32, 2.98MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.7M/862M [00:04<05:05, 2.64MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<09:36, 1.40MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:04<08:18, 1.62MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<06:12, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.8M/862M [00:06<07:01, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.2M/862M [00:06<06:27, 2.07MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.6M/862M [00:06<04:51, 2.75MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:08<06:14, 2.13MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:08<07:20, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:08<05:47, 2.29MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.4M/862M [00:08<04:17, 3.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<06:47, 1.95MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.5M/862M [00:10<06:19, 2.09MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.9M/862M [00:10<04:47, 2.75MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:12<06:09, 2.14MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.5M/862M [00:12<07:15, 1.81MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:12<05:45, 2.28MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.9M/862M [00:12<04:15, 3.09MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:14<06:58, 1.88MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.8M/862M [00:14<06:09, 2.13MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.9M/862M [00:14<04:40, 2.80MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.0M/862M [00:14<03:27, 3.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.7M/862M [00:16<12:30, 1.04MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:16<11:49, 1.10MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.5M/862M [00:16<08:53, 1.46MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.0M/862M [00:16<06:28, 2.00MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.8M/862M [00:18<08:00, 1.62MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.2M/862M [00:18<06:51, 1.89MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.9M/862M [00:18<05:18, 2.44MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:18<03:51, 3.34MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:20<12:49, 1.00MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.2M/862M [00:20<11:52, 1.09MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.9M/862M [00:20<08:59, 1.43MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.2M/862M [00:20<06:26, 1.99MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.2M/862M [00:22<10:46, 1.19MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.5M/862M [00:22<09:03, 1.42MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.9M/862M [00:22<06:38, 1.93MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:22<04:47, 2.66MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.3M/862M [00:24<1:52:37, 113kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.5M/862M [00:24<1:21:49, 156kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.2M/862M [00:24<57:50, 220kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 99.6M/862M [00:24<40:41, 312kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<31:38, 401kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<23:34, 538kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<16:48, 753kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<14:25, 874kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<13:06, 962kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<09:48, 1.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:28<07:05, 1.77MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<08:27, 1.48MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:22, 1.70MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:30<05:27, 2.29MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:30, 1.92MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:59, 2.08MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:32<04:32, 2.74MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:49, 2.13MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<06:57, 1.78MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<05:29, 2.26MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<04:04, 3.03MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:12, 1.99MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:45, 2.14MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<04:20, 2.83MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:39, 2.17MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<05:20, 2.29MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:01, 3.04MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:38<02:59, 4.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<22:24, 544kB/s] .vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<23:09, 526kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<18:01, 676kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<12:59, 936kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:40<09:13, 1.31MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<18:15, 664kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<16:17, 744kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<12:17, 985kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<08:48, 1.37MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:27, 1.15MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<10:50, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<08:19, 1.45MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:44<06:08, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:44<04:33, 2.63MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<17:19, 692kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<17:02, 703kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<13:06, 913kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<09:23, 1.27MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:46<06:52, 1.74MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:46, 1.11MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<12:02, 989kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<09:33, 1.25MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<06:56, 1.71MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:48<05:03, 2.34MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<20:59, 565kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<18:51, 628kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<14:20, 825kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<10:15, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:50<07:41, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<09:13, 1.28MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<12:50, 917kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<10:32, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<07:42, 1.53MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<05:38, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:52<04:16, 2.74MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<1:41:51, 115kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<1:14:19, 158kB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<52:37, 222kB/s]  .vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<37:00, 315kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:54<26:11, 445kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<24:45, 470kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<23:04, 504kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<17:37, 660kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<12:38, 918kB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:56<09:02, 1.28MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<12:47, 904kB/s] .vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<14:06, 820kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:58<11:16, 1.03MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:58<08:12, 1.41MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<05:56, 1.94MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<10:44, 1.07MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<13:10, 873kB/s] .vector_cache/glove.6B.zip:  20%|██        | 172M/862M [01:00<10:23, 1.11MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<07:40, 1.50MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<05:37, 2.04MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:00<04:12, 2.72MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<19:02, 600kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<18:26, 620kB/s].vector_cache/glove.6B.zip:  20%|██        | 177M/862M [01:02<13:58, 818kB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<10:10, 1.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:02<07:21, 1.55MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:02<05:24, 2.10MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<28:33, 398kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:04<25:04, 453kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<18:49, 604kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<13:29, 840kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<09:39, 1.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:06<17:21, 651kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<17:10, 658kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:06<13:16, 850kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<09:34, 1.18MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:06<06:53, 1.63MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<12:27, 901kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<13:43, 818kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:08<10:51, 1.03MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:52, 1.42MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<05:46, 1.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<08:28, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<10:54, 1.02MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:10<08:42, 1.28MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<06:27, 1.72MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<04:46, 2.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:10<03:35, 3.08MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<26:36, 417kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<23:36, 470kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:12<17:47, 623kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<12:46, 866kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<09:08, 1.21MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<16:54, 652kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:14<16:24, 671kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<12:40, 869kB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<09:11, 1.20MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<06:46, 1.62MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<09:14, 1.18MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:16<13:14, 826kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<11:00, 994kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<08:04, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<05:57, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:16<04:28, 2.43MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:27, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<09:58, 1.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<07:41, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<05:40, 1.91MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<04:16, 2.53MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:18<03:21, 3.23MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:19<12:37, 856kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<15:33, 695kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:20<12:27, 868kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<09:06, 1.18MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<06:38, 1.62MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<08:24, 1.28MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<11:47, 911kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:22<09:45, 1.10MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<07:13, 1.48MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<05:15, 2.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:22<04:04, 2.62MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<14:59, 712kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<16:20, 653kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:24<12:56, 825kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<09:22, 1.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:24<06:52, 1.55MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:24<05:05, 2.09MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<17:11, 617kB/s] .vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<17:48, 595kB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:26<13:56, 760kB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<10:04, 1.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<07:17, 1.45MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:26<05:22, 1.96MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<19:59, 527kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<19:47, 532kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<15:02, 700kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<10:53, 965kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:28<07:54, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:28<05:45, 1.82MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<28:08, 372kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<25:27, 411kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<18:58, 551kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<13:37, 767kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<09:49, 1.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:30<07:09, 1.45MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:42, 971kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:31<10:12, 1.02MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<07:44, 1.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<05:41, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<04:14, 2.44MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:33<07:27, 1.38MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<10:53, 948kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:57, 1.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<06:35, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:34<04:53, 2.10MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<03:38, 2.81MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<24:09, 425kB/s] .vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<19:33, 525kB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<14:14, 719kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<10:10, 1.01MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:36<07:26, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:36<05:45, 1.77MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<24:34, 415kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<25:37, 398kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<19:55, 511kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<14:23, 707kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:38<10:20, 982kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<07:37, 1.33MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:38<05:39, 1.79MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<18:42, 541kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<16:40, 607kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<12:27, 812kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<08:56, 1.13MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:40<06:44, 1.50MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<05:05, 1.98MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<08:25, 1.19MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<09:27, 1.06MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<07:26, 1.35MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:41<05:30, 1.82MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<04:06, 2.44MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:42<03:14, 3.08MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<15:31, 643kB/s] .vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<16:59, 588kB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<13:19, 749kB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<09:49, 1.02MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<07:10, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:44<05:21, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:20, 2.29MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<08:21, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<10:18, 961kB/s] .vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<08:19, 1.19MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:45<06:14, 1.59MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:45, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:46<03:43, 2.64MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<07:50, 1.25MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<09:34, 1.03MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<07:44, 1.27MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:47<05:48, 1.69MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:47<04:23, 2.23MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:21, 2.92MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:48<02:53, 3.38MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<1:53:36, 86.1kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<1:23:31, 117kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<59:15, 165kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<41:45, 234kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<29:38, 329kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:49<21:03, 462kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<15:05, 644kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<18:57, 512kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<16:56, 573kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<12:39, 767kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<09:10, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<06:49, 1.42MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:51<05:11, 1.86MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:52<03:55, 2.46MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<13:21, 721kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<12:59, 742kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<09:53, 974kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:53<07:17, 1.32MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<05:22, 1.79MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:53<04:11, 2.29MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:54<03:15, 2.94MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<7:32:50, 21.1kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<5:20:32, 29.9kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<3:45:08, 42.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:55<2:37:39, 60.5kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [01:55<1:50:25, 86.3kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:55<1:17:28, 123kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<1:02:46, 151kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<47:30, 200kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<34:00, 279kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<24:10, 392kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<17:12, 550kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [01:57<12:22, 764kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:57<08:57, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<07:31, 1.25MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<3:36:30, 43.6kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<2:36:41, 60.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<1:50:38, 85.2kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<1:17:53, 121kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<54:51, 171kB/s]  .vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [01:59<38:53, 242kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [01:59<27:32, 341kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:00<19:44, 475kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<19:57, 469kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<18:11, 515kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<13:50, 676kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:01<10:03, 928kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<07:19, 1.27MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:01<05:43, 1.63MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:02<04:22, 2.13MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<13:02, 713kB/s] .vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<13:18, 698kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<10:18, 901kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<07:36, 1.22MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:03<05:38, 1.64MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<04:17, 2.15MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<08:49, 1.04MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<09:56, 927kB/s] .vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<07:53, 1.17MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<05:51, 1.57MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:05<04:21, 2.11MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:05<03:34, 2.57MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<02:52, 3.18MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<3:28:57, 43.8kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<2:31:13, 60.5kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<1:47:05, 85.4kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<1:15:15, 121kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:07<53:04, 172kB/s]  .vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:07<37:35, 243kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<26:40, 341kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<19:11, 474kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<18:49, 483kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<17:38, 515kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<13:33, 670kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 318M/862M [02:09<09:52, 919kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<07:14, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:37, 1.61MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:09<04:16, 2.11MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:09<03:31, 2.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<11:20, 795kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<12:23, 728kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<09:37, 937kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<07:09, 1.26MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<05:25, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 323M/862M [02:11<04:12, 2.13MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:11<03:19, 2.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<02:43, 3.29MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<09:27, 946kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<11:01, 812kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<08:46, 1.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:33, 1.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:55, 1.81MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:49, 2.33MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:13<03:21, 2.65MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:13<02:47, 3.18MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<13:37, 651kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:15<12:27, 713kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<09:20, 950kB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<07:02, 1.26MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<05:18, 1.67MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:15<04:36, 1.92MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:39, 2.41MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:15<03:03, 2.89MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:17<10:29, 840kB/s] .vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<10:03, 876kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:17<07:38, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<05:42, 1.54MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<04:49, 1.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:46, 2.32MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<03:06, 2.82MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:17<02:42, 3.24MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<23:44, 368kB/s] .vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<19:02, 459kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<13:59, 624kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<10:15, 850kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<07:35, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 341M/862M [02:19<05:45, 1.51MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:19<04:29, 1.94MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<10:51, 799kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<09:57, 870kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<07:31, 1.15MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:21<05:39, 1.53MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<03:35, 2.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<02:55, 2.94MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:21<02:31, 3.40MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<10:25, 826kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<12:31, 687kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<09:56, 866kB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<07:21, 1.17MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<05:30, 1.56MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<04:31, 1.89MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:23<03:32, 2.42MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<02:54, 2.93MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<07:35, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:25<07:48, 1.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<06:06, 1.40MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<04:43, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<03:44, 2.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:25<03:02, 2.79MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:25<02:32, 3.34MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<10:49, 782kB/s] .vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<12:11, 695kB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:27<09:43, 870kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<07:17, 1.16MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<05:32, 1.52MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<04:17, 1.96MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:27<03:19, 2.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:27<02:58, 2.83MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<11:08, 754kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:28<10:04, 834kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<07:41, 1.09MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<05:48, 1.44MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:26, 1.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:29, 2.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:53, 2.88MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:29<02:28, 3.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<22:16, 374kB/s] .vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:30<20:09, 413kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<15:16, 545kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<11:08, 746kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<08:12, 1.01MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<06:05, 1.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<04:42, 1.76MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:41, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 366M/862M [02:32<14:33, 567kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<12:20, 669kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:33<09:14, 892kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:33<06:51, 1.20MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<05:11, 1.58MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:56, 2.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:33<03:17, 2.49MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<08:20, 983kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<09:55, 825kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:35<08:02, 1.02MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<06:03, 1.35MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<04:33, 1.79MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<03:36, 2.26MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:51, 2.85MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:35<02:27, 3.30MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<13:03, 622kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:36<13:10, 617kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:37<10:13, 794kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:37<07:34, 1.07MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<05:38, 1.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:37<04:16, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:19, 2.42MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<28:40, 281kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<23:43, 340kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<17:30, 460kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<12:34, 638kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<09:06, 880kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<06:35, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<05:04, 1.58MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<16:59, 470kB/s] .vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<15:11, 526kB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<11:27, 697kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:41<08:20, 955kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<06:05, 1.30MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<04:32, 1.75MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<10:34, 748kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:42<10:25, 760kB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<08:02, 984kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<05:57, 1.33MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<04:31, 1.74MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:46, 2.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:00, 2.62MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:29, 3.16MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<11:42, 670kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:44<13:06, 599kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<10:14, 766kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:45<07:33, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:45<05:36, 1.39MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:13, 1.85MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<03:25, 2.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:45<02:41, 2.90MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:46<42:40, 182kB/s] .vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<33:43, 231kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<24:34, 316kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:47<17:34, 442kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<12:37, 614kB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<09:08, 845kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<06:41, 1.15MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<28:41, 269kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<26:23, 292kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<20:22, 378kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:49<14:39, 525kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:49<10:35, 726kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<07:42, 995kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<05:46, 1.33MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<04:21, 1.75MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<09:55, 770kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<10:21, 738kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:51<08:05, 943kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:51<06:01, 1.26MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<04:29, 1.69MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<03:22, 2.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:51<02:50, 2.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<10:31, 720kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<10:29, 722kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:53<08:10, 926kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:53<06:02, 1.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<04:31, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<03:27, 2.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<06:26, 1.17MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:54<07:36, 986kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:55<06:08, 1.22MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:55<04:38, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:31, 2.12MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<02:45, 2.70MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<05:59, 1.24MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:56<07:30, 990kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<06:03, 1.23MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:57<04:31, 1.64MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<03:26, 2.15MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<02:42, 2.72MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<02:11, 3.35MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:58<07:01, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 420M/862M [02:58<08:13, 896kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<06:32, 1.12MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<04:53, 1.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<03:38, 2.01MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<02:55, 2.50MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:59<02:16, 3.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<10:34, 690kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:00<10:39, 684kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<08:07, 898kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<05:58, 1.22MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:01<04:30, 1.61MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<03:30, 2.06MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<02:44, 2.64MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:44, 1.26MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<07:00, 1.03MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<05:41, 1.27MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:03<04:14, 1.70MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<03:16, 2.20MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<02:32, 2.82MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<06:04, 1.18MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<07:12, 993kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<05:46, 1.24MB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:05<04:21, 1.64MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<03:19, 2.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<02:34, 2.75MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:56, 1.19MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<06:52, 1.03MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<05:31, 1.28MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<04:09, 1.70MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<03:11, 2.20MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<02:28, 2.84MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<06:02, 1.16MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<07:06, 987kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<05:35, 1.25MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:08<04:15, 1.64MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:13, 2.16MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:29, 2.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<06:48, 1.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<07:39, 908kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<05:56, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<04:26, 1.56MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:17, 2.10MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<02:38, 2.61MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:03, 3.35MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<5:24:50, 21.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<3:49:56, 29.9kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<2:41:22, 42.6kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<1:52:56, 60.7kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:12<1:19:16, 86.4kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<55:31, 123kB/s]   .vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<39:06, 174kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<36:02, 189kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:14<27:47, 245kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<19:56, 341kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<14:12, 478kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:14<10:13, 663kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<07:32, 898kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:16<07:28, 903kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<08:33, 788kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:49, 988kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:16<05:01, 1.34MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:47, 1.77MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<02:55, 2.29MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:17<02:19, 2.88MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:35, 1.19MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<06:53, 969kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<05:32, 1.20MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:18<04:09, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<03:08, 2.11MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:19<02:26, 2.70MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:52, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<06:50, 965kB/s] .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<05:26, 1.21MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<04:02, 1.63MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:20<03:01, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<02:23, 2.73MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:32, 1.44MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<05:39, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<04:34, 1.43MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:22<03:26, 1.89MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:22<02:36, 2.48MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<02:00, 3.22MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<08:39, 747kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:24<08:30, 760kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<06:31, 990kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:24<04:47, 1.35MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:24<03:33, 1.80MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<02:38, 2.42MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<24:13, 264kB/s] .vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:26<19:12, 333kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<13:54, 459kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<09:56, 642kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<07:11, 885kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:26<05:12, 1.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:26<03:51, 1.64MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<1:19:00, 80.1kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:28<57:21, 110kB/s]   .vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<40:34, 156kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:28<28:31, 221kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:28<20:13, 311kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<14:17, 439kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:30<14:28, 432kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<14:35, 429kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<11:20, 551kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:30<08:12, 761kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<05:53, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<04:23, 1.41MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:30<03:15, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<10:23, 596kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<09:18, 666kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:32<07:02, 878kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:32<05:07, 1.20MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:32<03:46, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:56, 1.24MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<05:35, 1.09MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:34<04:26, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<03:17, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<02:27, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<01:57, 3.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:43, 1.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<07:04, 857kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:36<05:36, 1.08MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:36<04:11, 1.44MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:13, 1.87MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<02:30, 2.40MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:36<02:01, 2.97MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:36<01:39, 3.61MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<08:56, 669kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<09:01, 663kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:38<06:58, 857kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:38<05:04, 1.17MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:38<03:46, 1.57MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:52, 2.06MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:38<02:14, 2.65MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<06:55, 855kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:40<07:16, 814kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:40<05:41, 1.04MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<04:11, 1.40MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:40<03:08, 1.87MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:40<02:24, 2.43MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<06:32, 895kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:42<06:47, 862kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<05:20, 1.09MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:42<03:55, 1.48MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:56, 1.98MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:42<02:16, 2.55MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:46, 3.25MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:44<55:36, 104kB/s] .vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<40:55, 141kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:44<29:03, 199kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:44<20:30, 281kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<14:33, 395kB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<10:20, 554kB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:44<07:25, 770kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<15:50, 360kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<13:14, 431kB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:46<09:46, 583kB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:46<07:01, 808kB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:46<05:05, 1.11MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:46<03:43, 1.51MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<1:32:19, 61.1kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<1:06:35, 84.7kB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:48<47:01, 120kB/s]   .vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:48<33:01, 170kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:48<23:12, 241kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<18:08, 307kB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:50<16:37, 335kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:50<12:39, 440kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:50<09:05, 611kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:50<06:30, 851kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<04:46, 1.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:50<03:28, 1.58MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<12:22, 445kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:52<10:36, 518kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:52<07:50, 701kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:52<05:40, 964kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<04:06, 1.33MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<03:08, 1.73MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<05:40, 957kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:54<05:52, 925kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:54<04:34, 1.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:54<03:23, 1.60MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:54<02:32, 2.12MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:54<01:56, 2.76MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<2:09:58, 41.3kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:56<1:32:51, 57.8kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:56<1:05:18, 82.0kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:56<45:47, 117kB/s]   .vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<32:04, 166kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:56<22:31, 235kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<29:37, 179kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<22:35, 234kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:58<16:15, 325kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [03:58<11:30, 458kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<08:12, 640kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:58<05:53, 889kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<07:31, 694kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<07:06, 734kB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:00<05:25, 961kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:00<03:57, 1.32MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:56, 1.76MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:00<02:14, 2.30MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:53, 1.06MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<05:14, 984kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:02<04:05, 1.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:02<03:01, 1.69MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:02<02:16, 2.25MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:22, 1.51MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<05:29, 926kB/s] .vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:04<04:42, 1.08MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:04<03:30, 1.45MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:04<02:36, 1.94MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<01:57, 2.57MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:19, 1.16MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<04:41, 1.07MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:06<03:42, 1.35MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:06<02:45, 1.81MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:03, 2.41MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:06<01:35, 3.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<19:57, 248kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<15:36, 317kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:08<11:18, 437kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:08<08:03, 611kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<05:45, 851kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<06:04, 804kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<07:14, 674kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<05:46, 845kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:10<04:16, 1.14MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<03:05, 1.56MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:20, 2.06MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<03:38, 1.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<04:07, 1.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:12<03:15, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:12<02:25, 1.97MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<01:55, 2.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<02:51, 1.66MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:56, 1.20MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<03:16, 1.44MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:14<02:28, 1.91MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<01:53, 2.48MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<01:27, 3.21MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:20, 1.08MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<04:56, 945kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:50, 1.21MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:16<02:54, 1.60MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:16<02:10, 2.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<01:41, 2.73MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:16<01:19, 3.49MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<10:13, 451kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<08:54, 517kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<06:38, 692kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:18<04:46, 959kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:18<03:29, 1.31MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:36, 1.74MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:37, 981kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<04:43, 959kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 591M/862M [04:19<03:41, 1.23MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:20<02:43, 1.66MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<02:01, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:09, 1.41MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<05:22, 830kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:24, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:21<03:17, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:22<02:27, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<02:00, 2.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:22<01:29, 2.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<16:30, 266kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<13:32, 325kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<09:55, 442kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:23<07:06, 616kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:24<05:10, 845kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<03:46, 1.15MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<02:49, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<02:08, 2.02MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<17:32, 247kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<14:13, 304kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<10:25, 415kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<07:26, 578kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<05:21, 800kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:54, 1.09MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:27<05:19, 800kB/s] .vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<05:28, 777kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<04:16, 996kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<03:09, 1.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:28<02:21, 1.79MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<01:47, 2.34MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:31, 926kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<04:45, 880kB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<03:44, 1.12MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:46, 1.50MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:30<02:03, 2.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<01:35, 2.60MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:33, 1.16MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<04:03, 1.02MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<03:14, 1.27MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:31<02:25, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:32<01:49, 2.23MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<01:24, 2.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:40, 1.10MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<04:06, 987kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<03:11, 1.27MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<02:22, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:33<01:49, 2.21MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:24, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<01:09, 3.45MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<06:04, 655kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<05:46, 689kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:35<04:22, 907kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<03:13, 1.23MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<02:22, 1.66MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:46, 2.20MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<01:30, 2.61MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<26:33, 147kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:37<20:43, 189kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<14:59, 261kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<10:39, 366kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<07:36, 510kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<05:29, 705kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<03:58, 969kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<05:51, 656kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:39<06:01, 639kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<04:37, 831kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<03:26, 1.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:39<02:33, 1.49MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:39<01:59, 1.91MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:40<01:32, 2.47MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<01:16, 2.97MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:41<03:32, 1.06MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:41<04:20, 872kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<03:25, 1.10MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:32, 1.48MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<02:01, 1.86MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:41<01:32, 2.41MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<01:13, 3.03MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:03, 3.53MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<1:19:42, 46.5kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<57:27, 64.5kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<40:34, 91.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<28:27, 130kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:43<19:59, 184kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<14:05, 260kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<11:22, 320kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<09:35, 380kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:45<07:02, 516kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:45<05:04, 714kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<03:44, 964kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:45, 1.30MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<02:04, 1.73MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:44, 754kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<04:54, 728kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:47<03:47, 941kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<02:49, 1.26MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<02:06, 1.68MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:47<01:38, 2.16MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:16, 2.76MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<01:02, 3.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<06:57, 503kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<06:26, 544kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:49<04:50, 721kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:49<03:34, 976kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:38, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:57, 1.77MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:49<01:33, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:49<01:13, 2.82MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<17:37, 195kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:51<13:52, 247kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<10:04, 340kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:51<07:11, 475kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:51<05:07, 662kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:51<03:42, 914kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<02:46, 1.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<05:05, 662kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:53<05:04, 662kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:53<03:55, 855kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:53, 1.16MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<02:09, 1.55MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:53<01:43, 1.93MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:53<01:20, 2.48MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<04:24, 746kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:55<04:33, 724kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:55<03:33, 925kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<02:37, 1.25MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [04:55<01:57, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:55<01:28, 2.19MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<01:11, 2.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:53, 661kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<04:51, 664kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:57<03:45, 857kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [04:57<02:45, 1.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<02:03, 1.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:57<01:37, 1.96MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:57<01:14, 2.55MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<2:15:31, 23.3kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<1:36:47, 32.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:59<1:08:08, 46.3kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<47:42, 65.8kB/s]  .vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<33:26, 93.6kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [04:59<23:29, 133kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [04:59<16:32, 188kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:00<11:41, 264kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 677M/862M [05:01<21:01, 147kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<15:32, 199kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:01<11:02, 279kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:01<07:51, 391kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:01<05:38, 542kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 679M/862M [05:01<04:04, 748kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:01<02:59, 1.01MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:15, 1.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<04:12, 718kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<04:25, 681kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:03<03:28, 869kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:03<02:34, 1.17MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<01:55, 1.55MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:03<01:27, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:03<01:07, 2.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<04:04, 725kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:05<04:18, 686kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:05<03:21, 878kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:05<02:27, 1.19MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:05<01:50, 1.58MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:05<01:25, 2.04MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:05<01:05, 2.63MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<02:55, 985kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:07<03:20, 864kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:07<02:38, 1.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:07<01:58, 1.45MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:29, 1.91MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:07<01:09, 2.46MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<02:07, 1.32MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:09<02:44, 1.03MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:09<02:13, 1.26MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:09<01:40, 1.67MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:15, 2.21MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:09<01:01, 2.69MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:09<00:48, 3.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<03:40, 747kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<03:41, 743kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:11<02:52, 952kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:11<02:06, 1.29MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:34, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:11<01:13, 2.20MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:11<00:56, 2.83MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<04:44, 564kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<04:24, 607kB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:13<03:21, 793kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<02:29, 1.07MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:13<01:53, 1.39MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:13<01:28, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<01:11, 2.20MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:13<00:57, 2.72MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:13, 810kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<03:03, 854kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:15<02:20, 1.11MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:15<01:46, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:23, 1.86MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:15<01:06, 2.31MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:15<00:54, 2.81MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:59, 1.27MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<02:07, 1.19MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:40, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:17<01:18, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:17<01:02, 2.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:51, 2.89MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:17<00:44, 3.37MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:14, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<02:14, 1.10MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:45, 1.40MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:19<01:21, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:19<01:04, 2.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 717M/862M [05:19<00:52, 2.78MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:19<00:43, 3.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:35, 924kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<02:28, 968kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:53, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:21<01:26, 1.65MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:21<01:09, 2.05MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:54, 2.59MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:21<00:45, 3.12MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:21<00:39, 3.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<02:26, 958kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 722M/862M [05:22<03:07, 747kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<02:29, 932kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:23<01:52, 1.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:23<01:25, 1.62MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<01:10, 1.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:23<00:54, 2.49MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:46, 2.95MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:23<00:46, 2.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<07:11, 315kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<05:53, 384kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:19, 521kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:25<03:19, 677kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<02:26, 920kB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:25<01:54, 1.16MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:32, 1.44MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:25<01:15, 1.76MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<01:05, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:25<00:56, 2.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:00, 1.09MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:55, 1.14MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<01:32, 1.42MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:15, 1.73MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:27<01:03, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 733M/862M [05:27<00:55, 2.36MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:47, 2.71MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:27<00:45, 2.80MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:40, 3.15MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:27<00:41, 3.06MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:27<00:37, 3.39MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<04:45, 447kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<03:47, 561kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<02:49, 750kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<02:08, 987kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:29<01:39, 1.26MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:29<01:19, 1.58MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:29<01:05, 1.91MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:54, 2.28MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:29<00:46, 2.64MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:29<00:45, 2.72MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<05:12, 395kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<03:40, 556kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:49, 723kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:31<02:09, 943kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:46, 1.15MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:26, 1.40MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:14, 1.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:31<01:05, 1.85MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:57, 2.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:52, 2.29MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:31<00:48, 2.45MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<00:46, 2.56MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:41, 1.18MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:27, 1.36MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:32<01:13, 1.62MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:02, 1.88MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:33<00:54, 2.15MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:50, 2.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:33<00:45, 2.57MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 746M/862M [05:33<00:40, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:44, 2.62MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:33<00:41, 2.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:37, 3.08MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:33<00:41, 2.75MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<04:29, 427kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<03:38, 527kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:44, 698kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<02:05, 911kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:35<01:37, 1.16MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:19, 1.43MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:35<01:05, 1.72MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:56, 2.00MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:35<00:49, 2.25MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:45, 2.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:35<00:42, 2.63MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:50, 653kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<02:37, 704kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<02:04, 891kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:40, 1.10MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:37<01:23, 1.32MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:10, 1.56MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<01:00, 1.82MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:55, 1.95MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:37<00:51, 2.11MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:47, 2.27MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:44, 2.42MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:37<00:43, 2.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:37<00:41, 2.62MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:37<00:41, 2.59MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<00:39, 2.75MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<01:55, 930kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:36, 1.11MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:18, 1.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:08, 1.54MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:04, 1.65MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:58, 1.80MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:55, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:50, 2.09MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:39<00:52, 1.98MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:54, 1.92MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:52, 2.00MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:46, 2.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:39<00:47, 2.17MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:47, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:43, 2.36MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<00:45, 2.28MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:51, 921kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:30, 1.13MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:14, 1.37MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:04, 1.58MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:57, 1.77MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:52, 1.93MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:41<00:45, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:49, 2.02MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:45, 2.20MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:45, 2.22MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:41<00:42, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:42, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 763M/862M [05:41<00:40, 2.46MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:41, 2.40MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:41<00:39, 2.51MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<01:42, 961kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:28, 1.11MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:13, 1.34MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<01:02, 1.58MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:42<00:54, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:43<00:48, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:43<00:43, 2.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:43, 2.24MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:39, 2.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:43<00:40, 2.36MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:38, 2.48MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:35, 2.65MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:37, 2.55MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:43<00:34, 2.78MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<02:16, 693kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:50, 852kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:26, 1.08MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:08, 1.37MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:03, 1.46MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<00:54, 1.72MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:46, 2.01MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:40, 2.27MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:40, 2.26MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:45<00:39, 2.32MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:37, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:42, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:42, 2.16MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:45<00:38, 2.33MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<00:42, 2.14MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<04:19, 349kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<03:17, 457kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:28, 604kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:54, 782kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:29, 997kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:11, 1.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:47<01:10, 1.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<01:02, 1.42MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:56, 1.58MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:54, 1.61MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:50, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:47<00:50, 1.74MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:47, 1.86MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:47, 1.83MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:44, 1.95MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:47<00:45, 1.90MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 775M/862M [05:48<00:42, 2.04MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:44, 1.93MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:48<00:44, 1.96MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:42, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:42, 2.04MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:42, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:40, 2.09MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:49<00:40, 2.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:39, 2.15MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:39, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:38, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:49<00:39, 2.13MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.16MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:36, 2.27MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:39, 2.12MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:49<00:38, 2.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:35, 2.32MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:39, 2.11MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<01:02, 1.33MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:53, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<00:54, 1.49MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:46, 1.75MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:45, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:41, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:48, 1.67MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:45, 1.77MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:41, 1.95MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:51<00:45, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:43, 1.84MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:39, 2.00MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:40, 1.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:51<00:37, 2.09MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:40, 1.92MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:38, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:36, 2.16MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:38, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:35, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:38, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:35, 2.20MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:37, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:36, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:33, 2.28MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:35, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:53<00:34, 2.19MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 787M/862M [05:53<00:38, 1.94MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:35, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:35, 2.13MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:41, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 787M/862M [05:53<00:39, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:42, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:42, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:40, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:42, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:39, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:43, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:54<00:39, 1.85MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:42, 1.72MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:41, 1.76MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:40, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:38, 1.86MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:38, 1.89MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:37, 1.91MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:55<00:37, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.95MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:55<00:36, 1.92MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:35, 2.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:34, 2.04MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:33, 2.07MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:56<00:33, 2.09MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:34, 2.01MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:31, 2.17MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:32, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:32, 2.12MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:30, 2.21MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:31, 2.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:57<00:30, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:30, 2.22MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:29, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:30, 2.23MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:57<00:28, 2.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:29, 2.28MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:57<00:29, 2.26MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:28, 2.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 796M/862M [05:58<00:29, 2.27MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:27, 2.41MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:27, 2.35MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:27, 2.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:26, 2.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:27, 2.36MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:25, 2.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:27, 2.37MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:59<00:25, 2.50MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:32, 1.93MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:30, 2.08MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [05:59<00:28, 2.22MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:26, 2.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:25, 2.42MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:59<00:24, 2.48MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:24, 2.54MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:23, 2.57MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:23, 2.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:22, 2.66MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:22, 2.71MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:21, 2.73MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<01:11, 836kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:59, 999kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:01<00:47, 1.23MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:39, 1.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:01<00:33, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:29, 1.98MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:26, 2.19MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:23, 2.45MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:22, 2.49MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:02<00:20, 2.69MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:21, 2.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:19, 2.86MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:20, 2.73MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:44, 528kB/s] .vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:21, 676kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:03<01:02, 872kB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:50, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:03<00:41, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 808M/862M [06:03<00:35, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:31, 1.71MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:32, 1.63MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:04<00:27, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:26, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:29, 1.77MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:27, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:26, 1.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<00:27, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:26, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:27, 1.87MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:34, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:05<00:31, 1.62MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:29, 1.74MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:27, 1.84MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:26, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:05<00:25, 1.98MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:24, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:23, 2.09MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:22, 2.22MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:06<00:24, 2.02MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:23, 2.07MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:22, 2.14MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:21, 2.20MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:22, 2.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:21, 2.23MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:21, 2.17MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:26, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:07<00:24, 1.91MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:23, 2.01MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:22, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:07<00:21, 2.14MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:20, 2.21MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.27MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:19, 2.32MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.36MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.40MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:17, 2.55MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:08<00:18, 2.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:17, 2.50MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:18, 2.33MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:08<00:32, 1.31MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:27, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:24, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:09<00:21, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:19, 2.12MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:18, 2.26MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:09<00:17, 2.38MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:16, 2.46MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:15, 2.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:15, 2.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:14, 2.65MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:14, 2.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:46, 834kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:38, 1.01MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:30, 1.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:11<00:25, 1.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:21, 1.74MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:18, 2.05MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:11<00:15, 2.32MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:15, 2.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.50MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:14, 2.53MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:12, 2.75MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:12, 2.78MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:12, 2.88MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:51, 674kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:41, 838kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:13<00:31, 1.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:25, 1.34MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:20, 1.60MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:13<00:17, 1.87MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 830M/862M [06:13<00:15, 2.13MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:13, 2.41MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:11, 2.70MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:11, 2.72MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:10, 3.02MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<00:10, 2.85MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<03:02, 167kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:14<02:13, 229kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 832M/862M [06:15<01:34, 317kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<01:07, 439kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:50, 583kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:15<00:36, 790kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:26, 1.05MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<00:21, 1.30MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:16, 1.67MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<00:12, 2.07MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:15<00:11, 2.33MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<13:32, 32.6kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<09:35, 45.8kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:17<06:39, 65.0kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<04:35, 92.3kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:17<03:08, 131kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<02:10, 185kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<01:31, 260kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<01:03, 364kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<00:44, 505kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:59, 377kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:18<00:47, 466kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:19<00:34, 636kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:19<00:24, 867kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:17, 1.18MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<00:13, 1.48MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:09, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:19<00:07, 2.45MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<04:23, 69.2kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:20<03:16, 92.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:21<02:18, 129kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:21<01:34, 183kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<01:04, 257kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:21<00:44, 360kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:30, 497kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:21, 686kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:24, 587kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:22<00:20, 667kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:22<00:15, 885kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 849M/862M [06:23<00:10, 1.18MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:23<00:07, 1.56MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:05, 2.05MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:04, 2.40MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:03, 2.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<01:00, 165kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:24<00:47, 209kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:24<00:33, 287kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:25<00:22, 402kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:25<00:14, 559kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:09, 775kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:06, 1.03MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:04, 1.39MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:04, 1.44MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:39, 148kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:26<00:29, 196kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:26<00:19, 274kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:27<00:12, 383kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:27<00:08, 524kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 858M/862M [06:27<00:05, 719kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 952kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.27MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:01, 1.56MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:00, 2.00MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:28<00:09, 182kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:06, 242kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:28<00:03, 335kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:29<00:00, 465kB/s].vector_cache/glove.6B.zip: 862MB [06:29, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<15:26:01,  7.20it/s]  0%|          | 903/400000 [00:00<10:46:58, 10.28it/s]  0%|          | 1804/400000 [00:00<7:32:04, 14.68it/s]  1%|          | 2705/400000 [00:00<5:15:57, 20.96it/s]  1%|          | 3603/400000 [00:00<3:40:53, 29.91it/s]  1%|          | 4471/400000 [00:00<2:34:30, 42.66it/s]  1%|▏         | 5381/400000 [00:00<1:48:07, 60.83it/s]  2%|▏         | 6277/400000 [00:00<1:15:44, 86.64it/s]  2%|▏         | 7213/400000 [00:00<53:06, 123.28it/s]   2%|▏         | 8134/400000 [00:01<37:17, 175.11it/s]  2%|▏         | 9065/400000 [00:01<26:15, 248.16it/s]  2%|▏         | 9967/400000 [00:01<18:33, 350.38it/s]  3%|▎         | 10883/400000 [00:01<13:10, 492.47it/s]  3%|▎         | 11808/400000 [00:01<09:24, 687.83it/s]  3%|▎         | 12729/400000 [00:01<06:46, 952.13it/s]  3%|▎         | 13641/400000 [00:01<04:56, 1301.80it/s]  4%|▎         | 14553/400000 [00:01<03:40, 1750.81it/s]  4%|▍         | 15460/400000 [00:01<02:46, 2308.13it/s]  4%|▍         | 16365/400000 [00:01<02:09, 2969.36it/s]  4%|▍         | 17269/400000 [00:02<01:42, 3718.43it/s]  5%|▍         | 18172/400000 [00:02<01:25, 4491.16it/s]  5%|▍         | 19066/400000 [00:02<01:12, 5272.43it/s]  5%|▍         | 19958/400000 [00:02<01:03, 6000.57it/s]  5%|▌         | 20865/400000 [00:02<00:56, 6677.99it/s]  5%|▌         | 21788/400000 [00:02<00:51, 7280.33it/s]  6%|▌         | 22696/400000 [00:02<00:48, 7738.83it/s]  6%|▌         | 23604/400000 [00:02<00:46, 8095.97it/s]  6%|▌         | 24510/400000 [00:02<00:45, 8343.23it/s]  6%|▋         | 25414/400000 [00:02<00:44, 8477.86it/s]  7%|▋         | 26329/400000 [00:03<00:43, 8667.59it/s]  7%|▋         | 27233/400000 [00:03<00:42, 8775.98it/s]  7%|▋         | 28136/400000 [00:03<00:42, 8814.85it/s]  7%|▋         | 29035/400000 [00:03<00:42, 8825.60it/s]  7%|▋         | 29930/400000 [00:03<00:41, 8848.71it/s]  8%|▊         | 30832/400000 [00:03<00:41, 8898.17it/s]  8%|▊         | 31728/400000 [00:03<00:41, 8903.55it/s]  8%|▊         | 32623/400000 [00:03<00:42, 8716.46it/s]  8%|▊         | 33506/400000 [00:03<00:41, 8748.67it/s]  9%|▊         | 34417/400000 [00:03<00:41, 8851.82it/s]  9%|▉         | 35322/400000 [00:04<00:40, 8908.37it/s]  9%|▉         | 36215/400000 [00:04<00:40, 8882.97it/s]  9%|▉         | 37105/400000 [00:04<00:41, 8757.85it/s]  9%|▉         | 37991/400000 [00:04<00:41, 8786.22it/s] 10%|▉         | 38873/400000 [00:04<00:41, 8795.22it/s] 10%|▉         | 39761/400000 [00:04<00:40, 8818.47it/s] 10%|█         | 40650/400000 [00:04<00:40, 8839.02it/s] 10%|█         | 41535/400000 [00:04<00:40, 8767.15it/s] 11%|█         | 42418/400000 [00:04<00:40, 8784.08it/s] 11%|█         | 43299/400000 [00:04<00:40, 8791.60it/s] 11%|█         | 44184/400000 [00:05<00:40, 8807.44it/s] 11%|█▏        | 45070/400000 [00:05<00:40, 8820.54it/s] 11%|█▏        | 45953/400000 [00:05<00:40, 8789.25it/s] 12%|█▏        | 46833/400000 [00:05<00:40, 8785.63it/s] 12%|█▏        | 47712/400000 [00:05<00:40, 8759.32it/s] 12%|█▏        | 48595/400000 [00:05<00:40, 8777.71it/s] 12%|█▏        | 49478/400000 [00:05<00:39, 8791.48it/s] 13%|█▎        | 50358/400000 [00:05<00:40, 8682.30it/s] 13%|█▎        | 51243/400000 [00:05<00:39, 8729.17it/s] 13%|█▎        | 52122/400000 [00:05<00:39, 8746.60it/s] 13%|█▎        | 53004/400000 [00:06<00:39, 8766.27it/s] 13%|█▎        | 53889/400000 [00:06<00:39, 8790.21it/s] 14%|█▎        | 54769/400000 [00:06<00:40, 8568.68it/s] 14%|█▍        | 55645/400000 [00:06<00:39, 8624.83it/s] 14%|█▍        | 56529/400000 [00:06<00:39, 8685.84it/s] 14%|█▍        | 57410/400000 [00:06<00:39, 8721.43it/s] 15%|█▍        | 58288/400000 [00:06<00:39, 8736.87it/s] 15%|█▍        | 59163/400000 [00:06<00:39, 8698.66it/s] 15%|█▌        | 60041/400000 [00:06<00:38, 8720.82it/s] 15%|█▌        | 60922/400000 [00:06<00:38, 8746.85it/s] 15%|█▌        | 61808/400000 [00:07<00:38, 8779.38it/s] 16%|█▌        | 62687/400000 [00:07<00:38, 8759.01it/s] 16%|█▌        | 63564/400000 [00:07<00:38, 8749.80it/s] 16%|█▌        | 64444/400000 [00:07<00:38, 8764.60it/s] 16%|█▋        | 65321/400000 [00:07<00:38, 8713.66it/s] 17%|█▋        | 66203/400000 [00:07<00:38, 8745.17it/s] 17%|█▋        | 67085/400000 [00:07<00:37, 8765.48it/s] 17%|█▋        | 67962/400000 [00:07<00:38, 8726.07it/s] 17%|█▋        | 68843/400000 [00:07<00:37, 8750.09it/s] 17%|█▋        | 69731/400000 [00:07<00:37, 8785.54it/s] 18%|█▊        | 70618/400000 [00:08<00:37, 8809.47it/s] 18%|█▊        | 71500/400000 [00:08<00:37, 8771.67it/s] 18%|█▊        | 72378/400000 [00:08<00:37, 8753.61it/s] 18%|█▊        | 73256/400000 [00:08<00:37, 8760.20it/s] 19%|█▊        | 74139/400000 [00:08<00:37, 8780.78it/s] 19%|█▉        | 75018/400000 [00:08<00:37, 8768.49it/s] 19%|█▉        | 75895/400000 [00:08<00:37, 8725.55it/s] 19%|█▉        | 76768/400000 [00:08<00:37, 8621.53it/s] 19%|█▉        | 77649/400000 [00:08<00:37, 8673.80it/s] 20%|█▉        | 78522/400000 [00:09<00:37, 8688.19it/s] 20%|█▉        | 79392/400000 [00:09<00:37, 8637.15it/s] 20%|██        | 80263/400000 [00:09<00:36, 8657.87it/s] 20%|██        | 81139/400000 [00:09<00:36, 8685.28it/s] 21%|██        | 82028/400000 [00:09<00:36, 8744.69it/s] 21%|██        | 82912/400000 [00:09<00:36, 8773.02it/s] 21%|██        | 83795/400000 [00:09<00:35, 8789.39it/s] 21%|██        | 84675/400000 [00:09<00:36, 8754.89it/s] 21%|██▏       | 85551/400000 [00:09<00:35, 8738.72it/s] 22%|██▏       | 86425/400000 [00:09<00:35, 8730.66it/s] 22%|██▏       | 87303/400000 [00:10<00:35, 8743.26it/s] 22%|██▏       | 88186/400000 [00:10<00:35, 8766.31it/s] 22%|██▏       | 89063/400000 [00:10<00:35, 8750.08it/s] 22%|██▏       | 89940/400000 [00:10<00:35, 8754.43it/s] 23%|██▎       | 90820/400000 [00:10<00:35, 8766.44it/s] 23%|██▎       | 91697/400000 [00:10<00:35, 8756.54it/s] 23%|██▎       | 92573/400000 [00:10<00:35, 8746.38it/s] 23%|██▎       | 93448/400000 [00:10<00:35, 8715.16it/s] 24%|██▎       | 94321/400000 [00:10<00:35, 8717.42it/s] 24%|██▍       | 95205/400000 [00:10<00:34, 8751.92it/s] 24%|██▍       | 96081/400000 [00:11<00:34, 8702.34it/s] 24%|██▍       | 96961/400000 [00:11<00:34, 8730.07it/s] 24%|██▍       | 97836/400000 [00:11<00:34, 8733.05it/s] 25%|██▍       | 98710/400000 [00:11<00:34, 8716.22it/s] 25%|██▍       | 99588/400000 [00:11<00:34, 8732.61it/s] 25%|██▌       | 100473/400000 [00:11<00:34, 8766.56it/s] 25%|██▌       | 101358/400000 [00:11<00:33, 8790.29it/s] 26%|██▌       | 102238/400000 [00:11<00:33, 8771.92it/s] 26%|██▌       | 103116/400000 [00:11<00:33, 8734.22it/s] 26%|██▌       | 103991/400000 [00:11<00:33, 8737.97it/s] 26%|██▌       | 104874/400000 [00:12<00:33, 8764.93it/s] 26%|██▋       | 105751/400000 [00:12<00:33, 8755.13it/s] 27%|██▋       | 106627/400000 [00:12<00:33, 8740.09it/s] 27%|██▋       | 107509/400000 [00:12<00:33, 8763.58it/s] 27%|██▋       | 108386/400000 [00:12<00:33, 8720.63it/s] 27%|██▋       | 109272/400000 [00:12<00:33, 8761.20it/s] 28%|██▊       | 110149/400000 [00:12<00:33, 8738.14it/s] 28%|██▊       | 111024/400000 [00:12<00:33, 8740.53it/s] 28%|██▊       | 111907/400000 [00:12<00:32, 8766.75it/s] 28%|██▊       | 112789/400000 [00:12<00:32, 8780.58it/s] 28%|██▊       | 113668/400000 [00:13<00:32, 8754.67it/s] 29%|██▊       | 114548/400000 [00:13<00:32, 8767.83it/s] 29%|██▉       | 115428/400000 [00:13<00:32, 8776.57it/s] 29%|██▉       | 116314/400000 [00:13<00:32, 8800.16it/s] 29%|██▉       | 117195/400000 [00:13<00:32, 8783.66it/s] 30%|██▉       | 118080/400000 [00:13<00:32, 8801.60it/s] 30%|██▉       | 118961/400000 [00:13<00:32, 8764.30it/s] 30%|██▉       | 119839/400000 [00:13<00:31, 8768.37it/s] 30%|███       | 120737/400000 [00:13<00:31, 8829.77it/s] 30%|███       | 121621/400000 [00:13<00:31, 8763.49it/s] 31%|███       | 122498/400000 [00:14<00:32, 8477.71it/s] 31%|███       | 123378/400000 [00:14<00:32, 8571.60it/s] 31%|███       | 124261/400000 [00:14<00:31, 8646.11it/s] 31%|███▏      | 125153/400000 [00:14<00:31, 8725.82it/s] 32%|███▏      | 126035/400000 [00:14<00:31, 8751.70it/s] 32%|███▏      | 126921/400000 [00:14<00:31, 8783.43it/s] 32%|███▏      | 127807/400000 [00:14<00:30, 8805.44it/s] 32%|███▏      | 128706/400000 [00:14<00:30, 8857.14it/s] 32%|███▏      | 129593/400000 [00:14<00:30, 8799.58it/s] 33%|███▎      | 130474/400000 [00:14<00:30, 8778.14it/s] 33%|███▎      | 131371/400000 [00:15<00:30, 8832.67it/s] 33%|███▎      | 132255/400000 [00:15<00:30, 8733.41it/s] 33%|███▎      | 133129/400000 [00:15<00:30, 8712.29it/s] 34%|███▎      | 134011/400000 [00:15<00:30, 8742.94it/s] 34%|███▎      | 134886/400000 [00:15<00:30, 8674.90it/s] 34%|███▍      | 135757/400000 [00:15<00:30, 8684.93it/s] 34%|███▍      | 136627/400000 [00:15<00:30, 8687.68it/s] 34%|███▍      | 137501/400000 [00:15<00:30, 8701.51it/s] 35%|███▍      | 138388/400000 [00:15<00:29, 8750.17it/s] 35%|███▍      | 139264/400000 [00:15<00:29, 8715.74it/s] 35%|███▌      | 140149/400000 [00:16<00:29, 8755.41it/s] 35%|███▌      | 141025/400000 [00:16<00:29, 8756.65it/s] 35%|███▌      | 141907/400000 [00:16<00:29, 8773.68it/s] 36%|███▌      | 142791/400000 [00:16<00:29, 8792.11it/s] 36%|███▌      | 143671/400000 [00:16<00:29, 8731.60it/s] 36%|███▌      | 144556/400000 [00:16<00:29, 8764.56it/s] 36%|███▋      | 145437/400000 [00:16<00:29, 8777.35it/s] 37%|███▋      | 146320/400000 [00:16<00:28, 8792.07it/s] 37%|███▋      | 147200/400000 [00:16<00:28, 8771.86it/s] 37%|███▋      | 148078/400000 [00:16<00:28, 8745.63it/s] 37%|███▋      | 148959/400000 [00:17<00:28, 8762.50it/s] 37%|███▋      | 149845/400000 [00:17<00:28, 8789.06it/s] 38%|███▊      | 150724/400000 [00:17<00:28, 8786.21it/s] 38%|███▊      | 151603/400000 [00:17<00:28, 8778.16it/s] 38%|███▊      | 152481/400000 [00:17<00:28, 8734.12it/s] 38%|███▊      | 153355/400000 [00:17<00:28, 8707.56it/s] 39%|███▊      | 154235/400000 [00:17<00:28, 8732.53it/s] 39%|███▉      | 155120/400000 [00:17<00:27, 8766.07it/s] 39%|███▉      | 156005/400000 [00:17<00:27, 8788.34it/s] 39%|███▉      | 156884/400000 [00:17<00:27, 8754.39it/s] 39%|███▉      | 157766/400000 [00:18<00:27, 8771.62it/s] 40%|███▉      | 158644/400000 [00:18<00:27, 8732.60it/s] 40%|███▉      | 159520/400000 [00:18<00:27, 8739.42it/s] 40%|████      | 160406/400000 [00:18<00:27, 8772.99it/s] 40%|████      | 161289/400000 [00:18<00:27, 8789.15it/s] 41%|████      | 162174/400000 [00:18<00:27, 8805.18it/s] 41%|████      | 163075/400000 [00:18<00:26, 8864.69it/s] 41%|████      | 163971/400000 [00:18<00:26, 8891.50it/s] 41%|████      | 164861/400000 [00:18<00:26, 8849.79it/s] 41%|████▏     | 165747/400000 [00:18<00:26, 8759.67it/s] 42%|████▏     | 166624/400000 [00:19<00:26, 8759.35it/s] 42%|████▏     | 167510/400000 [00:19<00:26, 8788.97it/s] 42%|████▏     | 168401/400000 [00:19<00:26, 8822.44it/s] 42%|████▏     | 169284/400000 [00:19<00:26, 8821.87it/s] 43%|████▎     | 170167/400000 [00:19<00:26, 8805.15it/s] 43%|████▎     | 171048/400000 [00:19<00:26, 8760.58it/s] 43%|████▎     | 171932/400000 [00:19<00:25, 8782.38it/s] 43%|████▎     | 172811/400000 [00:19<00:25, 8774.87it/s] 43%|████▎     | 173695/400000 [00:19<00:25, 8793.92it/s] 44%|████▎     | 174582/400000 [00:19<00:25, 8814.14it/s] 44%|████▍     | 175464/400000 [00:20<00:25, 8734.18it/s] 44%|████▍     | 176358/400000 [00:20<00:25, 8793.50it/s] 44%|████▍     | 177241/400000 [00:20<00:25, 8803.85it/s] 45%|████▍     | 178124/400000 [00:20<00:25, 8810.13it/s] 45%|████▍     | 179010/400000 [00:20<00:25, 8823.35it/s] 45%|████▍     | 179893/400000 [00:20<00:25, 8615.40it/s] 45%|████▌     | 180756/400000 [00:20<00:25, 8532.46it/s] 45%|████▌     | 181628/400000 [00:20<00:25, 8586.33it/s] 46%|████▌     | 182506/400000 [00:20<00:25, 8640.82it/s] 46%|████▌     | 183377/400000 [00:20<00:25, 8660.20it/s] 46%|████▌     | 184260/400000 [00:21<00:24, 8709.78it/s] 46%|████▋     | 185132/400000 [00:21<00:24, 8700.40it/s] 47%|████▋     | 186015/400000 [00:21<00:24, 8736.16it/s] 47%|████▋     | 186889/400000 [00:21<00:24, 8734.41it/s] 47%|████▋     | 187763/400000 [00:21<00:24, 8706.75it/s] 47%|████▋     | 188634/400000 [00:21<00:24, 8666.81it/s] 47%|████▋     | 189509/400000 [00:21<00:24, 8690.76it/s] 48%|████▊     | 190384/400000 [00:21<00:24, 8707.04it/s] 48%|████▊     | 191262/400000 [00:21<00:23, 8727.18it/s] 48%|████▊     | 192138/400000 [00:21<00:23, 8735.58it/s] 48%|████▊     | 193012/400000 [00:22<00:23, 8704.89it/s] 48%|████▊     | 193887/400000 [00:22<00:23, 8716.97it/s] 49%|████▊     | 194766/400000 [00:22<00:23, 8738.29it/s] 49%|████▉     | 195644/400000 [00:22<00:23, 8749.72it/s] 49%|████▉     | 196520/400000 [00:22<00:23, 8721.54it/s] 49%|████▉     | 197393/400000 [00:22<00:23, 8698.17it/s] 50%|████▉     | 198274/400000 [00:22<00:23, 8729.87it/s] 50%|████▉     | 199158/400000 [00:22<00:22, 8760.30it/s] 50%|█████     | 200035/400000 [00:22<00:22, 8752.54it/s] 50%|█████     | 200918/400000 [00:22<00:22, 8774.83it/s] 50%|█████     | 201801/400000 [00:23<00:22, 8789.95it/s] 51%|█████     | 202684/400000 [00:23<00:22, 8799.60it/s] 51%|█████     | 203570/400000 [00:23<00:22, 8816.24it/s] 51%|█████     | 204457/400000 [00:23<00:22, 8830.92it/s] 51%|█████▏    | 205341/400000 [00:23<00:22, 8815.53it/s] 52%|█████▏    | 206223/400000 [00:23<00:22, 8799.66it/s] 52%|█████▏    | 207104/400000 [00:23<00:21, 8800.11it/s] 52%|█████▏    | 207988/400000 [00:23<00:21, 8811.68it/s] 52%|█████▏    | 208879/400000 [00:23<00:21, 8838.59it/s] 52%|█████▏    | 209763/400000 [00:24<00:21, 8787.32it/s] 53%|█████▎    | 210643/400000 [00:24<00:21, 8789.78it/s] 53%|█████▎    | 211551/400000 [00:24<00:21, 8873.03it/s] 53%|█████▎    | 212446/400000 [00:24<00:21, 8893.73it/s] 53%|█████▎    | 213348/400000 [00:24<00:20, 8930.05it/s] 54%|█████▎    | 214243/400000 [00:24<00:20, 8935.10it/s] 54%|█████▍    | 215138/400000 [00:24<00:20, 8937.05it/s] 54%|█████▍    | 216032/400000 [00:24<00:20, 8930.86it/s] 54%|█████▍    | 216927/400000 [00:24<00:20, 8935.96it/s] 54%|█████▍    | 217821/400000 [00:24<00:20, 8871.14it/s] 55%|█████▍    | 218709/400000 [00:25<00:20, 8798.71it/s] 55%|█████▍    | 219595/400000 [00:25<00:20, 8814.91it/s] 55%|█████▌    | 220480/400000 [00:25<00:20, 8822.57it/s] 55%|█████▌    | 221383/400000 [00:25<00:20, 8883.52it/s] 56%|█████▌    | 222272/400000 [00:25<00:20, 8856.30it/s] 56%|█████▌    | 223158/400000 [00:25<00:19, 8847.30it/s] 56%|█████▌    | 224043/400000 [00:25<00:19, 8842.10it/s] 56%|█████▌    | 224935/400000 [00:25<00:19, 8863.93it/s] 56%|█████▋    | 225822/400000 [00:25<00:19, 8859.79it/s] 57%|█████▋    | 226709/400000 [00:25<00:19, 8799.62it/s] 57%|█████▋    | 227604/400000 [00:26<00:19, 8843.26it/s] 57%|█████▋    | 228510/400000 [00:26<00:19, 8905.15it/s] 57%|█████▋    | 229418/400000 [00:26<00:19, 8956.71it/s] 58%|█████▊    | 230314/400000 [00:26<00:18, 8936.99it/s] 58%|█████▊    | 231208/400000 [00:26<00:18, 8908.78it/s] 58%|█████▊    | 232100/400000 [00:26<00:18, 8886.42it/s] 58%|█████▊    | 232989/400000 [00:26<00:18, 8886.09it/s] 58%|█████▊    | 233878/400000 [00:26<00:18, 8873.87it/s] 59%|█████▊    | 234766/400000 [00:26<00:18, 8843.88it/s] 59%|█████▉    | 235654/400000 [00:26<00:18, 8853.86it/s] 59%|█████▉    | 236540/400000 [00:27<00:18, 8845.50it/s] 59%|█████▉    | 237429/400000 [00:27<00:18, 8856.14it/s] 60%|█████▉    | 238363/400000 [00:27<00:17, 8993.59it/s] 60%|█████▉    | 239263/400000 [00:27<00:18, 8903.20it/s] 60%|██████    | 240154/400000 [00:27<00:18, 8852.03it/s] 60%|██████    | 241057/400000 [00:27<00:17, 8904.13it/s] 60%|██████    | 241948/400000 [00:27<00:17, 8873.17it/s] 61%|██████    | 242843/400000 [00:27<00:17, 8895.75it/s] 61%|██████    | 243735/400000 [00:27<00:17, 8902.31it/s] 61%|██████    | 244635/400000 [00:27<00:17, 8929.91it/s] 61%|██████▏   | 245546/400000 [00:28<00:17, 8981.28it/s] 62%|██████▏   | 246445/400000 [00:28<00:17, 8961.31it/s] 62%|██████▏   | 247375/400000 [00:28<00:16, 9057.61it/s] 62%|██████▏   | 248282/400000 [00:28<00:16, 9056.34it/s] 62%|██████▏   | 249188/400000 [00:28<00:16, 9037.68it/s] 63%|██████▎   | 250092/400000 [00:28<00:16, 8985.02it/s] 63%|██████▎   | 250991/400000 [00:28<00:16, 8920.16it/s] 63%|██████▎   | 251884/400000 [00:28<00:16, 8873.63it/s] 63%|██████▎   | 252776/400000 [00:28<00:16, 8885.43it/s] 63%|██████▎   | 253670/400000 [00:28<00:16, 8900.35it/s] 64%|██████▎   | 254561/400000 [00:29<00:16, 8887.07it/s] 64%|██████▍   | 255452/400000 [00:29<00:16, 8893.56it/s] 64%|██████▍   | 256342/400000 [00:29<00:16, 8886.00it/s] 64%|██████▍   | 257231/400000 [00:29<00:16, 8835.38it/s] 65%|██████▍   | 258115/400000 [00:29<00:16, 8780.98it/s] 65%|██████▍   | 258995/400000 [00:29<00:16, 8786.03it/s] 65%|██████▍   | 259874/400000 [00:29<00:15, 8774.44it/s] 65%|██████▌   | 260752/400000 [00:29<00:15, 8748.98it/s] 65%|██████▌   | 261634/400000 [00:29<00:15, 8770.12it/s] 66%|██████▌   | 262522/400000 [00:29<00:15, 8801.09it/s] 66%|██████▌   | 263405/400000 [00:30<00:15, 8808.61it/s] 66%|██████▌   | 264289/400000 [00:30<00:15, 8814.87it/s] 66%|██████▋   | 265171/400000 [00:30<00:15, 8694.73it/s] 67%|██████▋   | 266053/400000 [00:30<00:15, 8729.31it/s] 67%|██████▋   | 266927/400000 [00:30<00:15, 8621.95it/s] 67%|██████▋   | 267808/400000 [00:30<00:15, 8675.11it/s] 67%|██████▋   | 268677/400000 [00:30<00:15, 8677.90it/s] 67%|██████▋   | 269550/400000 [00:30<00:15, 8690.94it/s] 68%|██████▊   | 270431/400000 [00:30<00:14, 8725.14it/s] 68%|██████▊   | 271308/400000 [00:30<00:14, 8736.20it/s] 68%|██████▊   | 272192/400000 [00:31<00:14, 8764.68it/s] 68%|██████▊   | 273076/400000 [00:31<00:14, 8785.16it/s] 68%|██████▊   | 273955/400000 [00:31<00:14, 8751.10it/s] 69%|██████▊   | 274839/400000 [00:31<00:14, 8775.52it/s] 69%|██████▉   | 275722/400000 [00:31<00:14, 8790.37it/s] 69%|██████▉   | 276606/400000 [00:31<00:14, 8804.36it/s] 69%|██████▉   | 277487/400000 [00:31<00:13, 8790.97it/s] 70%|██████▉   | 278367/400000 [00:31<00:13, 8788.81it/s] 70%|██████▉   | 279249/400000 [00:31<00:13, 8796.74it/s] 70%|███████   | 280129/400000 [00:31<00:13, 8751.93it/s] 70%|███████   | 281014/400000 [00:32<00:13, 8778.29it/s] 70%|███████   | 281892/400000 [00:32<00:13, 8761.15it/s] 71%|███████   | 282771/400000 [00:32<00:13, 8767.68it/s] 71%|███████   | 283649/400000 [00:32<00:13, 8771.28it/s] 71%|███████   | 284528/400000 [00:32<00:13, 8775.80it/s] 71%|███████▏  | 285406/400000 [00:32<00:13, 8752.98it/s] 72%|███████▏  | 286282/400000 [00:32<00:13, 8720.60it/s] 72%|███████▏  | 287155/400000 [00:32<00:12, 8682.18it/s] 72%|███████▏  | 288034/400000 [00:32<00:12, 8713.77it/s] 72%|███████▏  | 288909/400000 [00:32<00:12, 8721.89it/s] 72%|███████▏  | 289785/400000 [00:33<00:12, 8732.28it/s] 73%|███████▎  | 290659/400000 [00:33<00:12, 8666.94it/s] 73%|███████▎  | 291538/400000 [00:33<00:12, 8696.37it/s] 73%|███████▎  | 292411/400000 [00:33<00:12, 8704.55it/s] 73%|███████▎  | 293305/400000 [00:33<00:12, 8772.20it/s] 74%|███████▎  | 294193/400000 [00:33<00:12, 8802.81it/s] 74%|███████▍  | 295095/400000 [00:33<00:11, 8865.35it/s] 74%|███████▍  | 295982/400000 [00:33<00:11, 8830.22it/s] 74%|███████▍  | 296866/400000 [00:33<00:11, 8807.16it/s] 74%|███████▍  | 297747/400000 [00:33<00:11, 8788.75it/s] 75%|███████▍  | 298626/400000 [00:34<00:11, 8783.36it/s] 75%|███████▍  | 299505/400000 [00:34<00:11, 8723.15it/s] 75%|███████▌  | 300379/400000 [00:34<00:11, 8725.23it/s] 75%|███████▌  | 301259/400000 [00:34<00:11, 8745.18it/s] 76%|███████▌  | 302143/400000 [00:34<00:11, 8771.46it/s] 76%|███████▌  | 303021/400000 [00:34<00:11, 8764.79it/s] 76%|███████▌  | 303901/400000 [00:34<00:10, 8773.31it/s] 76%|███████▌  | 304779/400000 [00:34<00:10, 8766.04it/s] 76%|███████▋  | 305665/400000 [00:34<00:10, 8792.71it/s] 77%|███████▋  | 306548/400000 [00:34<00:10, 8801.88it/s] 77%|███████▋  | 307429/400000 [00:35<00:10, 8775.66it/s] 77%|███████▋  | 308307/400000 [00:35<00:10, 8744.68it/s] 77%|███████▋  | 309182/400000 [00:35<00:10, 8745.47it/s] 78%|███████▊  | 310069/400000 [00:35<00:10, 8779.64it/s] 78%|███████▊  | 310950/400000 [00:35<00:10, 8788.71it/s] 78%|███████▊  | 311830/400000 [00:35<00:10, 8791.91it/s] 78%|███████▊  | 312710/400000 [00:35<00:09, 8778.26it/s] 78%|███████▊  | 313588/400000 [00:35<00:09, 8742.14it/s] 79%|███████▊  | 314463/400000 [00:35<00:09, 8658.99it/s] 79%|███████▉  | 315332/400000 [00:35<00:09, 8665.83it/s] 79%|███████▉  | 316202/400000 [00:36<00:09, 8674.31it/s] 79%|███████▉  | 317084/400000 [00:36<00:09, 8716.65it/s] 79%|███████▉  | 317956/400000 [00:36<00:09, 8703.38it/s] 80%|███████▉  | 318840/400000 [00:36<00:09, 8742.32it/s] 80%|███████▉  | 319719/400000 [00:36<00:09, 8754.83it/s] 80%|████████  | 320595/400000 [00:36<00:09, 8737.81it/s] 80%|████████  | 321469/400000 [00:36<00:08, 8738.41it/s] 81%|████████  | 322343/400000 [00:36<00:08, 8710.87it/s] 81%|████████  | 323218/400000 [00:36<00:08, 8720.54it/s] 81%|████████  | 324099/400000 [00:36<00:08, 8746.64it/s] 81%|████████  | 324974/400000 [00:37<00:08, 8728.60it/s] 81%|████████▏ | 325860/400000 [00:37<00:08, 8767.54it/s] 82%|████████▏ | 326737/400000 [00:37<00:08, 8715.35it/s] 82%|████████▏ | 327616/400000 [00:37<00:08, 8735.87it/s] 82%|████████▏ | 328500/400000 [00:37<00:08, 8765.08it/s] 82%|████████▏ | 329377/400000 [00:37<00:08, 8644.72it/s] 83%|████████▎ | 330260/400000 [00:37<00:08, 8698.71it/s] 83%|████████▎ | 331138/400000 [00:37<00:07, 8721.95it/s] 83%|████████▎ | 332021/400000 [00:37<00:07, 8751.89it/s] 83%|████████▎ | 332897/400000 [00:37<00:07, 8671.31it/s] 83%|████████▎ | 333780/400000 [00:38<00:07, 8718.29it/s] 84%|████████▎ | 334653/400000 [00:38<00:07, 8686.85it/s] 84%|████████▍ | 335531/400000 [00:38<00:07, 8711.89it/s] 84%|████████▍ | 336411/400000 [00:38<00:07, 8736.79it/s] 84%|████████▍ | 337286/400000 [00:38<00:07, 8739.73it/s] 85%|████████▍ | 338172/400000 [00:38<00:07, 8772.60it/s] 85%|████████▍ | 339055/400000 [00:38<00:06, 8787.03it/s] 85%|████████▍ | 339940/400000 [00:38<00:06, 8804.49it/s] 85%|████████▌ | 340821/400000 [00:38<00:06, 8792.53it/s] 85%|████████▌ | 341701/400000 [00:38<00:06, 8767.03it/s] 86%|████████▌ | 342581/400000 [00:39<00:06, 8776.84it/s] 86%|████████▌ | 343461/400000 [00:39<00:06, 8782.72it/s] 86%|████████▌ | 344340/400000 [00:39<00:06, 8779.34it/s] 86%|████████▋ | 345218/400000 [00:39<00:06, 8743.21it/s] 87%|████████▋ | 346098/400000 [00:39<00:06, 8758.37it/s] 87%|████████▋ | 346974/400000 [00:39<00:06, 8706.24it/s] 87%|████████▋ | 347848/400000 [00:39<00:05, 8714.57it/s] 87%|████████▋ | 348721/400000 [00:39<00:05, 8717.41it/s] 87%|████████▋ | 349593/400000 [00:39<00:05, 8701.31it/s] 88%|████████▊ | 350469/400000 [00:39<00:05, 8718.64it/s] 88%|████████▊ | 351352/400000 [00:40<00:05, 8750.27it/s] 88%|████████▊ | 352236/400000 [00:40<00:05, 8775.17it/s] 88%|████████▊ | 353118/400000 [00:40<00:05, 8786.69it/s] 88%|████████▊ | 353997/400000 [00:40<00:05, 8729.61it/s] 89%|████████▊ | 354871/400000 [00:40<00:05, 8722.03it/s] 89%|████████▉ | 355757/400000 [00:40<00:05, 8761.38it/s] 89%|████████▉ | 356640/400000 [00:40<00:04, 8780.49it/s] 89%|████████▉ | 357524/400000 [00:40<00:04, 8797.89it/s] 90%|████████▉ | 358404/400000 [00:40<00:04, 8722.97it/s] 90%|████████▉ | 359277/400000 [00:41<00:04, 8664.40it/s] 90%|█████████ | 360159/400000 [00:41<00:04, 8709.18it/s] 90%|█████████ | 361042/400000 [00:41<00:04, 8743.68it/s] 90%|█████████ | 361926/400000 [00:41<00:04, 8770.02it/s] 91%|█████████ | 362804/400000 [00:41<00:04, 8761.78it/s] 91%|█████████ | 363689/400000 [00:41<00:04, 8786.85it/s] 91%|█████████ | 364584/400000 [00:41<00:04, 8833.32it/s] 91%|█████████▏| 365493/400000 [00:41<00:03, 8907.68it/s] 92%|█████████▏| 366385/400000 [00:41<00:03, 8887.18it/s] 92%|█████████▏| 367274/400000 [00:41<00:03, 8843.24it/s] 92%|█████████▏| 368159/400000 [00:42<00:03, 8833.57it/s] 92%|█████████▏| 369043/400000 [00:42<00:03, 8824.50it/s] 92%|█████████▏| 369926/400000 [00:42<00:03, 8824.97it/s] 93%|█████████▎| 370829/400000 [00:42<00:03, 8883.58it/s] 93%|█████████▎| 371718/400000 [00:42<00:03, 8836.36it/s] 93%|█████████▎| 372602/400000 [00:42<00:03, 8836.65it/s] 93%|█████████▎| 373486/400000 [00:42<00:03, 8822.60it/s] 94%|█████████▎| 374369/400000 [00:42<00:02, 8704.37it/s] 94%|█████████▍| 375252/400000 [00:42<00:02, 8738.90it/s] 94%|█████████▍| 376127/400000 [00:42<00:02, 8739.07it/s] 94%|█████████▍| 377002/400000 [00:43<00:02, 8657.83it/s] 94%|█████████▍| 377884/400000 [00:43<00:02, 8703.70it/s] 95%|█████████▍| 378768/400000 [00:43<00:02, 8742.08it/s] 95%|█████████▍| 379650/400000 [00:43<00:02, 8763.18it/s] 95%|█████████▌| 380547/400000 [00:43<00:02, 8823.59it/s] 95%|█████████▌| 381451/400000 [00:43<00:02, 8885.76it/s] 96%|█████████▌| 382340/400000 [00:43<00:01, 8866.13it/s] 96%|█████████▌| 383227/400000 [00:43<00:01, 8840.36it/s] 96%|█████████▌| 384112/400000 [00:43<00:01, 8816.27it/s] 96%|█████████▌| 384994/400000 [00:43<00:01, 8777.14it/s] 96%|█████████▋| 385877/400000 [00:44<00:01, 8791.83it/s] 97%|█████████▋| 386762/400000 [00:44<00:01, 8806.46it/s] 97%|█████████▋| 387644/400000 [00:44<00:01, 8810.35it/s] 97%|█████████▋| 388526/400000 [00:44<00:01, 8790.69it/s] 97%|█████████▋| 389406/400000 [00:44<00:01, 8696.91it/s] 98%|█████████▊| 390288/400000 [00:44<00:01, 8732.45it/s] 98%|█████████▊| 391172/400000 [00:44<00:01, 8762.65it/s] 98%|█████████▊| 392053/400000 [00:44<00:00, 8774.38it/s] 98%|█████████▊| 392939/400000 [00:44<00:00, 8797.71it/s] 98%|█████████▊| 393824/400000 [00:44<00:00, 8811.86it/s] 99%|█████████▊| 394706/400000 [00:45<00:00, 8806.15it/s] 99%|█████████▉| 395591/400000 [00:45<00:00, 8818.98it/s] 99%|█████████▉| 396473/400000 [00:45<00:00, 8815.89it/s] 99%|█████████▉| 397389/400000 [00:45<00:00, 8915.12it/s]100%|█████████▉| 398281/400000 [00:45<00:00, 8873.30it/s]100%|█████████▉| 399190/400000 [00:45<00:00, 8935.35it/s]100%|█████████▉| 399999/400000 [00:45<00:00, 8767.86it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f36c6043ba8>, <torchtext.data.dataset.TabularDataset object at 0x7f36c6043cf8>, <torchtext.vocab.Vocab object at 0x7f36c6043c18>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.36 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 12.42 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 12.42 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.22 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.22 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.11 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.11 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.07 file/s]2020-06-20 06:18:02.661639: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-20 06:18:02.666023: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-06-20 06:18:02.666176: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5606689cfd70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-20 06:18:02.666190: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 34%|███▍      | 3391488/9912422 [00:00<00:00, 33850186.76it/s]9920512it [00:00, 33963506.80it/s]                             
0it [00:00, ?it/s]32768it [00:00, 557412.43it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 455860.67it/s]1654784it [00:00, 11781974.89it/s]                         
0it [00:00, ?it/s]8192it [00:00, 195808.76it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '2675f1e090030e6958e45c46c6313291532e6ed8', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/2675f1e090030e6958e45c46c6313291532e6ed8

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/2675f1e090030e6958e45c46c6313291532e6ed8

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fb964e32620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fb964e32620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7fb9d03f90d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7fb9d03f90d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7fb9ea125ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7fb9ea125ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb97dc74950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb97dc74950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb97dc74950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 487791.93it/s] 90%|█████████ | 8945664/9912422 [00:00<00:01, 695165.71it/s]9920512it [00:00, 47200869.34it/s]                           
0it [00:00, ?it/s]32768it [00:00, 578181.53it/s]
0it [00:00, ?it/s]  3%|▎         | 57344/1648877 [00:00<00:02, 564118.60it/s] 76%|███████▌  | 1253376/1648877 [00:00<00:00, 789493.45it/s]1654784it [00:00, 6947566.08it/s]                            
0it [00:00, ?it/s]8192it [00:00, 252353.81it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb97b28ab00>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb97b280da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7fb97dc74598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7fb97dc74598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7fb97dc74598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:58,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:57,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:57,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:56,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:56,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:56,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:55,  2.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.90 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:38,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:38,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:38,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:36,  3.90 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:25,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:24,  5.48 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.66 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:16,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.66 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:10, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:10, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:10, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:10, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:10, 10.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 14.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:00<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:00<00:05, 19.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:00<00:05, 19.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:00<00:05, 19.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.15 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 33.15 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 41.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 50.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 50.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 59.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 68.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 68.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:01<00:00, 72.29 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 79.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:01<00:00, 79.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:01<00:00, 84.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:01<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:01<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:01<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:01<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:01<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.00s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.00s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.00s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 84.26 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.00s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 84.26 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 39.70 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.08s/ url]
0 examples [00:00, ? examples/s]2020-06-10 18:09:02.812456: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-10 18:09:02.827143: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-06-10 18:09:02.827312: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55b541a58340 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-10 18:09:02.827328: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
45 examples [00:00, 448.13 examples/s]156 examples [00:00, 545.20 examples/s]266 examples [00:00, 641.92 examples/s]375 examples [00:00, 732.22 examples/s]489 examples [00:00, 819.15 examples/s]604 examples [00:00, 894.82 examples/s]712 examples [00:00, 941.47 examples/s]820 examples [00:00, 977.81 examples/s]928 examples [00:00, 1004.56 examples/s]1038 examples [00:01, 1029.50 examples/s]1151 examples [00:01, 1056.89 examples/s]1266 examples [00:01, 1083.05 examples/s]1379 examples [00:01, 1094.06 examples/s]1491 examples [00:01, 1100.22 examples/s]1602 examples [00:01, 1095.31 examples/s]1713 examples [00:01, 1091.93 examples/s]1823 examples [00:01, 1059.07 examples/s]1934 examples [00:01, 1071.29 examples/s]2042 examples [00:01, 1064.55 examples/s]2149 examples [00:02, 1026.15 examples/s]2258 examples [00:02, 1043.08 examples/s]2366 examples [00:02, 1053.65 examples/s]2472 examples [00:02, 1048.92 examples/s]2584 examples [00:02, 1067.31 examples/s]2691 examples [00:02, 1062.15 examples/s]2798 examples [00:02, 1045.31 examples/s]2904 examples [00:02, 1048.97 examples/s]3010 examples [00:02, 1026.20 examples/s]3115 examples [00:02, 1030.48 examples/s]3219 examples [00:03, 1018.02 examples/s]3327 examples [00:03, 1034.12 examples/s]3437 examples [00:03, 1050.26 examples/s]3547 examples [00:03, 1061.55 examples/s]3654 examples [00:03, 1049.78 examples/s]3763 examples [00:03, 1058.94 examples/s]3873 examples [00:03, 1069.61 examples/s]3981 examples [00:03, 1065.77 examples/s]4088 examples [00:03, 1056.83 examples/s]4198 examples [00:03, 1069.13 examples/s]4306 examples [00:04, 1070.27 examples/s]4418 examples [00:04, 1083.59 examples/s]4530 examples [00:04, 1092.99 examples/s]4640 examples [00:04, 1090.40 examples/s]4750 examples [00:04, 1088.48 examples/s]4860 examples [00:04, 1091.25 examples/s]4970 examples [00:04, 1077.03 examples/s]5082 examples [00:04, 1087.40 examples/s]5194 examples [00:04, 1094.44 examples/s]5306 examples [00:04, 1100.08 examples/s]5417 examples [00:05, 1071.44 examples/s]5525 examples [00:05, 1070.71 examples/s]5633 examples [00:05, 1063.44 examples/s]5742 examples [00:05, 1070.64 examples/s]5851 examples [00:05, 1073.83 examples/s]5959 examples [00:05, 1051.01 examples/s]6069 examples [00:05, 1064.81 examples/s]6182 examples [00:05, 1081.59 examples/s]6291 examples [00:05, 1067.86 examples/s]6399 examples [00:06, 1070.41 examples/s]6510 examples [00:06, 1079.29 examples/s]6619 examples [00:06, 1061.19 examples/s]6733 examples [00:06, 1083.17 examples/s]6845 examples [00:06, 1091.44 examples/s]6956 examples [00:06, 1095.52 examples/s]7069 examples [00:06, 1105.21 examples/s]7180 examples [00:06, 1098.52 examples/s]7290 examples [00:06, 1095.10 examples/s]7400 examples [00:06, 1048.79 examples/s]7507 examples [00:07, 1054.79 examples/s]7614 examples [00:07, 1058.97 examples/s]7721 examples [00:07, 1050.88 examples/s]7827 examples [00:07, 1053.15 examples/s]7933 examples [00:07, 1049.99 examples/s]8039 examples [00:07, 981.19 examples/s] 8148 examples [00:07, 1007.97 examples/s]8257 examples [00:07, 1030.98 examples/s]8361 examples [00:07, 1019.42 examples/s]8471 examples [00:07, 1040.98 examples/s]8579 examples [00:08, 1051.62 examples/s]8690 examples [00:08, 1066.64 examples/s]8798 examples [00:08, 1068.78 examples/s]8906 examples [00:08, 1043.39 examples/s]9019 examples [00:08, 1066.95 examples/s]9127 examples [00:08, 1068.44 examples/s]9238 examples [00:08, 1079.50 examples/s]9347 examples [00:08, 1081.76 examples/s]9456 examples [00:08, 1073.05 examples/s]9566 examples [00:09, 1079.32 examples/s]9675 examples [00:09, 1076.45 examples/s]9787 examples [00:09, 1088.91 examples/s]9896 examples [00:09, 1084.70 examples/s]10005 examples [00:09, 1022.93 examples/s]10115 examples [00:09, 1044.36 examples/s]10223 examples [00:09, 1054.59 examples/s]10334 examples [00:09, 1068.57 examples/s]10443 examples [00:09, 1073.83 examples/s]10551 examples [00:09, 1058.15 examples/s]10658 examples [00:10, 1047.13 examples/s]10766 examples [00:10, 1054.76 examples/s]10878 examples [00:10, 1071.65 examples/s]10986 examples [00:10, 1071.95 examples/s]11095 examples [00:10, 1077.11 examples/s]11206 examples [00:10, 1084.64 examples/s]11316 examples [00:10, 1086.80 examples/s]11428 examples [00:10, 1095.43 examples/s]11541 examples [00:10, 1103.95 examples/s]11652 examples [00:10, 1071.27 examples/s]11762 examples [00:11, 1077.20 examples/s]11870 examples [00:11, 1077.64 examples/s]11982 examples [00:11, 1089.02 examples/s]12094 examples [00:11, 1096.55 examples/s]12205 examples [00:11, 1099.11 examples/s]12317 examples [00:11, 1103.52 examples/s]12428 examples [00:11, 1101.93 examples/s]12539 examples [00:11, 1097.08 examples/s]12654 examples [00:11, 1110.46 examples/s]12766 examples [00:11, 1049.85 examples/s]12879 examples [00:12, 1072.03 examples/s]12987 examples [00:12, 1062.70 examples/s]13099 examples [00:12, 1077.46 examples/s]13209 examples [00:12, 1082.17 examples/s]13321 examples [00:12, 1092.27 examples/s]13435 examples [00:12, 1105.17 examples/s]13546 examples [00:12, 1098.85 examples/s]13664 examples [00:12, 1120.20 examples/s]13778 examples [00:12, 1123.76 examples/s]13891 examples [00:12, 1105.92 examples/s]14008 examples [00:13, 1121.97 examples/s]14121 examples [00:13, 1122.62 examples/s]14234 examples [00:13, 1122.54 examples/s]14347 examples [00:13, 1096.06 examples/s]14465 examples [00:13, 1119.93 examples/s]14578 examples [00:13, 1116.31 examples/s]14690 examples [00:13, 1107.71 examples/s]14804 examples [00:13, 1114.98 examples/s]14916 examples [00:13, 1108.52 examples/s]15029 examples [00:14, 1113.67 examples/s]15144 examples [00:14, 1123.26 examples/s]15257 examples [00:14, 1096.11 examples/s]15367 examples [00:14, 1094.19 examples/s]15481 examples [00:14, 1106.73 examples/s]15598 examples [00:14, 1124.92 examples/s]15713 examples [00:14, 1129.63 examples/s]15827 examples [00:14, 1109.84 examples/s]15939 examples [00:14, 1101.48 examples/s]16050 examples [00:14, 1084.23 examples/s]16159 examples [00:15, 1074.40 examples/s]16272 examples [00:15, 1090.02 examples/s]16384 examples [00:15, 1096.30 examples/s]16497 examples [00:15, 1104.24 examples/s]16608 examples [00:15, 1096.76 examples/s]16720 examples [00:15, 1103.26 examples/s]16831 examples [00:15, 1104.69 examples/s]16942 examples [00:15, 1064.75 examples/s]17049 examples [00:15, 1051.82 examples/s]17155 examples [00:15, 1048.88 examples/s]17261 examples [00:16, 1016.79 examples/s]17369 examples [00:16, 1033.86 examples/s]17476 examples [00:16, 1042.24 examples/s]17588 examples [00:16, 1062.62 examples/s]17699 examples [00:16, 1075.07 examples/s]17810 examples [00:16, 1084.34 examples/s]17919 examples [00:16, 1085.90 examples/s]18028 examples [00:16, 1080.86 examples/s]18137 examples [00:16, 1081.73 examples/s]18246 examples [00:16, 1041.28 examples/s]18359 examples [00:17, 1064.47 examples/s]18469 examples [00:17, 1072.43 examples/s]18577 examples [00:17, 1056.84 examples/s]18685 examples [00:17, 1060.87 examples/s]18794 examples [00:17, 1067.81 examples/s]18901 examples [00:17, 1032.23 examples/s]19005 examples [00:17, 1033.95 examples/s]19120 examples [00:17, 1066.08 examples/s]19228 examples [00:17, 1062.37 examples/s]19336 examples [00:18, 1065.99 examples/s]19444 examples [00:18, 1068.10 examples/s]19551 examples [00:18, 1053.79 examples/s]19662 examples [00:18, 1067.54 examples/s]19772 examples [00:18, 1074.15 examples/s]19884 examples [00:18, 1086.13 examples/s]19994 examples [00:18, 1087.41 examples/s]20103 examples [00:18, 993.89 examples/s] 20214 examples [00:18, 1026.08 examples/s]20319 examples [00:18, 1027.42 examples/s]20427 examples [00:19, 1039.08 examples/s]20532 examples [00:19, 1021.99 examples/s]20635 examples [00:19, 1018.51 examples/s]20738 examples [00:19, 1016.71 examples/s]20840 examples [00:19, 1015.24 examples/s]20945 examples [00:19, 1023.34 examples/s]21056 examples [00:19, 1046.83 examples/s]21161 examples [00:19, 1012.07 examples/s]21263 examples [00:19, 1013.92 examples/s]21373 examples [00:19, 1035.84 examples/s]21485 examples [00:20, 1058.43 examples/s]21596 examples [00:20, 1071.02 examples/s]21704 examples [00:20, 1035.86 examples/s]21816 examples [00:20, 1057.90 examples/s]21923 examples [00:20, 1059.80 examples/s]22031 examples [00:20, 1063.16 examples/s]22142 examples [00:20, 1075.75 examples/s]22250 examples [00:20, 1074.22 examples/s]22358 examples [00:20, 1038.80 examples/s]22463 examples [00:21, 1014.19 examples/s]22567 examples [00:21, 1021.49 examples/s]22670 examples [00:21, 1010.38 examples/s]22772 examples [00:21, 1003.52 examples/s]22882 examples [00:21, 1029.70 examples/s]22993 examples [00:21, 1050.20 examples/s]23102 examples [00:21, 1060.35 examples/s]23213 examples [00:21, 1072.11 examples/s]23321 examples [00:21, 1020.96 examples/s]23428 examples [00:21, 1033.01 examples/s]23532 examples [00:22, 1009.01 examples/s]23634 examples [00:22, 1003.24 examples/s]23735 examples [00:22, 989.27 examples/s] 23835 examples [00:22, 991.84 examples/s]23944 examples [00:22, 1019.06 examples/s]24055 examples [00:22, 1044.03 examples/s]24166 examples [00:22, 1061.81 examples/s]24278 examples [00:22, 1076.32 examples/s]24386 examples [00:22, 1061.68 examples/s]24496 examples [00:22, 1070.24 examples/s]24604 examples [00:23, 1073.14 examples/s]24714 examples [00:23, 1078.55 examples/s]24822 examples [00:23, 1068.00 examples/s]24930 examples [00:23, 1070.10 examples/s]25038 examples [00:23, 1063.68 examples/s]25154 examples [00:23, 1090.27 examples/s]25268 examples [00:23, 1102.32 examples/s]25379 examples [00:23, 1102.39 examples/s]25490 examples [00:23, 1090.54 examples/s]25600 examples [00:23, 1081.99 examples/s]25709 examples [00:24, 1062.29 examples/s]25821 examples [00:24, 1078.09 examples/s]25929 examples [00:24, 1059.83 examples/s]26040 examples [00:24, 1072.48 examples/s]26148 examples [00:24, 1072.78 examples/s]26258 examples [00:24, 1080.42 examples/s]26372 examples [00:24, 1095.02 examples/s]26482 examples [00:24, 1093.07 examples/s]26595 examples [00:24, 1102.55 examples/s]26707 examples [00:25, 1106.75 examples/s]26818 examples [00:25, 1096.72 examples/s]26928 examples [00:25, 1068.38 examples/s]27036 examples [00:25, 1012.69 examples/s]27140 examples [00:25, 1020.22 examples/s]27250 examples [00:25, 1041.16 examples/s]27362 examples [00:25, 1062.33 examples/s]27474 examples [00:25, 1077.10 examples/s]27583 examples [00:25, 1075.50 examples/s]27695 examples [00:25, 1087.36 examples/s]27807 examples [00:26, 1094.33 examples/s]27918 examples [00:26, 1097.56 examples/s]28030 examples [00:26, 1103.66 examples/s]28141 examples [00:26, 1037.76 examples/s]28250 examples [00:26, 1052.74 examples/s]28362 examples [00:26, 1070.81 examples/s]28470 examples [00:26, 1064.82 examples/s]28581 examples [00:26, 1077.85 examples/s]28690 examples [00:26, 1065.14 examples/s]28797 examples [00:26, 1053.08 examples/s]28910 examples [00:27, 1074.64 examples/s]29019 examples [00:27, 1078.50 examples/s]29128 examples [00:27, 1052.37 examples/s]29234 examples [00:27, 1047.01 examples/s]29346 examples [00:27, 1066.67 examples/s]29456 examples [00:27, 1074.29 examples/s]29566 examples [00:27, 1078.22 examples/s]29674 examples [00:27, 1076.31 examples/s]29782 examples [00:27, 1032.93 examples/s]29886 examples [00:28, 1033.71 examples/s]30000 examples [00:28, 1062.79 examples/s]30107 examples [00:28, 1008.77 examples/s]30216 examples [00:28, 1025.78 examples/s]30320 examples [00:28, 977.75 examples/s] 30419 examples [00:28, 954.33 examples/s]30524 examples [00:28, 980.68 examples/s]30626 examples [00:28, 990.25 examples/s]30726 examples [00:28, 977.70 examples/s]30834 examples [00:28, 1004.58 examples/s]30947 examples [00:29, 1037.02 examples/s]31058 examples [00:29, 1057.01 examples/s]31166 examples [00:29, 1063.41 examples/s]31275 examples [00:29, 1070.92 examples/s]31383 examples [00:29, 1047.51 examples/s]31489 examples [00:29, 1026.53 examples/s]31594 examples [00:29, 1032.08 examples/s]31698 examples [00:29, 1029.79 examples/s]31805 examples [00:29, 1040.49 examples/s]31917 examples [00:29, 1061.68 examples/s]32028 examples [00:30, 1075.03 examples/s]32136 examples [00:30, 1069.48 examples/s]32244 examples [00:30, 1049.32 examples/s]32355 examples [00:30, 1064.51 examples/s]32465 examples [00:30, 1074.17 examples/s]32578 examples [00:30, 1090.05 examples/s]32693 examples [00:30, 1105.59 examples/s]32806 examples [00:30, 1110.62 examples/s]32919 examples [00:30, 1115.82 examples/s]33031 examples [00:31, 1101.78 examples/s]33145 examples [00:31, 1111.93 examples/s]33258 examples [00:31, 1114.38 examples/s]33370 examples [00:31, 1101.85 examples/s]33481 examples [00:31, 1038.84 examples/s]33586 examples [00:31, 1023.97 examples/s]33695 examples [00:31, 1041.03 examples/s]33803 examples [00:31, 1050.25 examples/s]33916 examples [00:31, 1072.51 examples/s]34024 examples [00:31, 1070.49 examples/s]34132 examples [00:32, 1058.89 examples/s]34244 examples [00:32, 1076.00 examples/s]34354 examples [00:32, 1081.20 examples/s]34463 examples [00:32, 1077.70 examples/s]34571 examples [00:32, 1073.77 examples/s]34679 examples [00:32, 1073.01 examples/s]34787 examples [00:32, 1043.57 examples/s]34897 examples [00:32, 1059.78 examples/s]35008 examples [00:32, 1074.25 examples/s]35116 examples [00:32, 1061.77 examples/s]35223 examples [00:33, 1048.86 examples/s]35329 examples [00:33, 1038.83 examples/s]35439 examples [00:33, 1054.19 examples/s]35551 examples [00:33, 1072.96 examples/s]35660 examples [00:33, 1077.40 examples/s]35768 examples [00:33, 1043.82 examples/s]35877 examples [00:33, 1056.02 examples/s]35983 examples [00:33, 1055.19 examples/s]36095 examples [00:33, 1071.97 examples/s]36207 examples [00:33, 1084.43 examples/s]36321 examples [00:34, 1098.57 examples/s]36433 examples [00:34, 1103.09 examples/s]36544 examples [00:34, 1087.75 examples/s]36653 examples [00:34, 1074.64 examples/s]36761 examples [00:34, 1058.91 examples/s]36868 examples [00:34, 1042.08 examples/s]36978 examples [00:34, 1056.75 examples/s]37086 examples [00:34, 1063.29 examples/s]37197 examples [00:34, 1076.55 examples/s]37310 examples [00:35, 1091.15 examples/s]37426 examples [00:35, 1110.42 examples/s]37543 examples [00:35, 1125.17 examples/s]37656 examples [00:35, 1110.52 examples/s]37771 examples [00:35, 1120.11 examples/s]37884 examples [00:35, 1099.06 examples/s]37995 examples [00:35, 1083.69 examples/s]38104 examples [00:35, 1070.35 examples/s]38219 examples [00:35, 1090.67 examples/s]38329 examples [00:35, 1061.47 examples/s]38438 examples [00:36, 1069.21 examples/s]38546 examples [00:36, 1027.08 examples/s]38657 examples [00:36, 1048.76 examples/s]38763 examples [00:36, 1041.92 examples/s]38868 examples [00:36, 996.98 examples/s] 38969 examples [00:36, 984.09 examples/s]39077 examples [00:36, 1010.16 examples/s]39184 examples [00:36, 1027.33 examples/s]39288 examples [00:36, 1022.39 examples/s]39393 examples [00:36, 1027.51 examples/s]39496 examples [00:37, 1024.16 examples/s]39599 examples [00:37, 1020.98 examples/s]39710 examples [00:37, 1044.63 examples/s]39815 examples [00:37, 1034.26 examples/s]39919 examples [00:37, 1030.73 examples/s]40023 examples [00:37, 919.75 examples/s] 40131 examples [00:37, 961.39 examples/s]40237 examples [00:37, 988.23 examples/s]40340 examples [00:37, 999.72 examples/s]40442 examples [00:38, 1003.51 examples/s]40544 examples [00:38, 1008.33 examples/s]40647 examples [00:38, 1014.65 examples/s]40749 examples [00:38, 1003.90 examples/s]40858 examples [00:38, 1027.90 examples/s]40962 examples [00:38, 1020.71 examples/s]41069 examples [00:38, 1034.96 examples/s]41180 examples [00:38, 1054.32 examples/s]41286 examples [00:38, 1054.96 examples/s]41400 examples [00:38, 1079.00 examples/s]41509 examples [00:39, 1064.04 examples/s]41617 examples [00:39, 1066.08 examples/s]41724 examples [00:39, 1065.85 examples/s]41835 examples [00:39, 1077.80 examples/s]41947 examples [00:39, 1088.13 examples/s]42056 examples [00:39, 1085.79 examples/s]42168 examples [00:39, 1094.92 examples/s]42278 examples [00:39, 1056.45 examples/s]42391 examples [00:39, 1077.22 examples/s]42507 examples [00:39, 1098.58 examples/s]42621 examples [00:40, 1110.18 examples/s]42737 examples [00:40, 1122.97 examples/s]42850 examples [00:40, 1123.38 examples/s]42963 examples [00:40, 1124.69 examples/s]43076 examples [00:40, 1052.74 examples/s]43183 examples [00:40, 1052.70 examples/s]43290 examples [00:40, 1056.69 examples/s]43404 examples [00:40, 1080.11 examples/s]43513 examples [00:40, 1072.50 examples/s]43624 examples [00:41, 1082.55 examples/s]43738 examples [00:41, 1097.26 examples/s]43849 examples [00:41, 1100.82 examples/s]43961 examples [00:41, 1105.50 examples/s]44072 examples [00:41, 1104.76 examples/s]44183 examples [00:41, 1094.48 examples/s]44293 examples [00:41, 1074.82 examples/s]44401 examples [00:41, 1075.87 examples/s]44513 examples [00:41, 1086.79 examples/s]44622 examples [00:41, 1074.94 examples/s]44730 examples [00:42, 1033.61 examples/s]44839 examples [00:42, 1049.77 examples/s]44946 examples [00:42, 1053.65 examples/s]45052 examples [00:42, 1036.39 examples/s]45158 examples [00:42, 1041.32 examples/s]45263 examples [00:42, 1027.78 examples/s]45370 examples [00:42, 1039.99 examples/s]45483 examples [00:42, 1064.37 examples/s]45595 examples [00:42, 1077.81 examples/s]45706 examples [00:42, 1085.93 examples/s]45816 examples [00:43, 1088.52 examples/s]45925 examples [00:43, 1031.35 examples/s]46029 examples [00:43, 1000.26 examples/s]46136 examples [00:43, 1020.13 examples/s]46246 examples [00:43, 1042.51 examples/s]46351 examples [00:43, 1035.16 examples/s]46455 examples [00:43, 1025.50 examples/s]46561 examples [00:43, 1035.45 examples/s]46673 examples [00:43, 1059.42 examples/s]46780 examples [00:44, 1044.12 examples/s]46885 examples [00:44, 1042.00 examples/s]46999 examples [00:44, 1067.85 examples/s]47112 examples [00:44, 1084.28 examples/s]47221 examples [00:44, 1078.73 examples/s]47332 examples [00:44, 1087.72 examples/s]47441 examples [00:44, 1047.69 examples/s]47547 examples [00:44, 1041.93 examples/s]47657 examples [00:44, 1052.72 examples/s]47763 examples [00:44, 1012.28 examples/s]47870 examples [00:45, 1027.31 examples/s]47982 examples [00:45, 1051.01 examples/s]48100 examples [00:45, 1084.27 examples/s]48214 examples [00:45, 1098.31 examples/s]48325 examples [00:45, 1083.55 examples/s]48438 examples [00:45, 1094.49 examples/s]48549 examples [00:45, 1098.83 examples/s]48660 examples [00:45, 1101.39 examples/s]48771 examples [00:45, 1095.71 examples/s]48883 examples [00:45, 1102.45 examples/s]48996 examples [00:46, 1109.69 examples/s]49108 examples [00:46, 1076.68 examples/s]49216 examples [00:46, 1068.37 examples/s]49327 examples [00:46, 1078.31 examples/s]49439 examples [00:46, 1089.20 examples/s]49549 examples [00:46, 1084.89 examples/s]49658 examples [00:46, 1077.41 examples/s]49766 examples [00:46, 1050.38 examples/s]49874 examples [00:46, 1058.53 examples/s]49983 examples [00:46, 1066.03 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7049/50000 [00:00<00:00, 70486.37 examples/s] 41%|████▏     | 20626/50000 [00:00<00:00, 82361.43 examples/s] 69%|██████▉   | 34707/50000 [00:00<00:00, 94075.47 examples/s] 96%|█████████▌| 48123/50000 [00:00<00:00, 103338.09 examples/s]                                                                0 examples [00:00, ? examples/s]92 examples [00:00, 917.75 examples/s]205 examples [00:00, 972.30 examples/s]311 examples [00:00, 994.59 examples/s]420 examples [00:00, 1020.31 examples/s]531 examples [00:00, 1042.96 examples/s]640 examples [00:00, 1055.29 examples/s]749 examples [00:00, 1063.14 examples/s]849 examples [00:00, 1022.29 examples/s]947 examples [00:00, 1005.83 examples/s]1045 examples [00:01, 947.91 examples/s]1152 examples [00:01, 979.75 examples/s]1250 examples [00:01, 976.63 examples/s]1352 examples [00:01, 986.72 examples/s]1451 examples [00:01, 780.27 examples/s]1550 examples [00:01, 832.03 examples/s]1661 examples [00:01, 897.79 examples/s]1766 examples [00:01, 938.28 examples/s]1875 examples [00:01, 979.03 examples/s]1977 examples [00:02, 980.82 examples/s]2078 examples [00:02, 968.43 examples/s]2186 examples [00:02, 996.96 examples/s]2288 examples [00:02, 983.99 examples/s]2388 examples [00:02, 978.82 examples/s]2487 examples [00:02, 968.79 examples/s]2592 examples [00:02, 990.59 examples/s]2701 examples [00:02, 1017.45 examples/s]2804 examples [00:02, 989.12 examples/s] 2911 examples [00:02, 1011.94 examples/s]3020 examples [00:03, 1031.72 examples/s]3125 examples [00:03, 1036.64 examples/s]3234 examples [00:03, 1050.34 examples/s]3340 examples [00:03, 1033.58 examples/s]3444 examples [00:03, 1034.75 examples/s]3554 examples [00:03, 1052.39 examples/s]3660 examples [00:03, 1043.54 examples/s]3773 examples [00:03, 1067.74 examples/s]3881 examples [00:03, 1067.28 examples/s]3997 examples [00:03, 1093.28 examples/s]4117 examples [00:04, 1121.57 examples/s]4232 examples [00:04, 1129.78 examples/s]4346 examples [00:04, 1132.80 examples/s]4460 examples [00:04, 1128.43 examples/s]4574 examples [00:04, 1131.72 examples/s]4691 examples [00:04, 1140.87 examples/s]4806 examples [00:04, 1121.85 examples/s]4919 examples [00:04, 1116.27 examples/s]5031 examples [00:04, 1102.34 examples/s]5142 examples [00:04, 1088.11 examples/s]5251 examples [00:05, 1084.97 examples/s]5360 examples [00:05, 1075.93 examples/s]5468 examples [00:05, 1027.02 examples/s]5572 examples [00:05, 1029.54 examples/s]5676 examples [00:05, 1006.96 examples/s]5785 examples [00:05, 1029.84 examples/s]5889 examples [00:05, 1029.99 examples/s]6003 examples [00:05, 1058.33 examples/s]6111 examples [00:05, 1061.98 examples/s]6223 examples [00:06, 1077.78 examples/s]6332 examples [00:06, 1030.06 examples/s]6440 examples [00:06, 1044.06 examples/s]6546 examples [00:06, 1045.44 examples/s]6653 examples [00:06, 1052.67 examples/s]6760 examples [00:06, 1056.19 examples/s]6866 examples [00:06, 1028.48 examples/s]6975 examples [00:06, 1043.55 examples/s]7080 examples [00:06, 1026.12 examples/s]7188 examples [00:06, 1041.44 examples/s]7298 examples [00:07, 1057.15 examples/s]7405 examples [00:07, 1059.22 examples/s]7514 examples [00:07, 1066.90 examples/s]7621 examples [00:07, 1064.47 examples/s]7728 examples [00:07, 1061.44 examples/s]7835 examples [00:07, 1028.78 examples/s]7947 examples [00:07, 1054.23 examples/s]8060 examples [00:07, 1075.46 examples/s]8168 examples [00:07, 1048.78 examples/s]8278 examples [00:07, 1062.13 examples/s]8392 examples [00:08, 1082.40 examples/s]8504 examples [00:08, 1093.01 examples/s]8617 examples [00:08, 1101.77 examples/s]8728 examples [00:08, 1099.41 examples/s]8839 examples [00:08, 1073.59 examples/s]8952 examples [00:08, 1089.01 examples/s]9064 examples [00:08, 1097.79 examples/s]9177 examples [00:08, 1106.69 examples/s]9288 examples [00:08, 1095.47 examples/s]9399 examples [00:09, 1097.97 examples/s]9512 examples [00:09, 1105.33 examples/s]9623 examples [00:09, 1091.59 examples/s]9733 examples [00:09, 1054.31 examples/s]9839 examples [00:09, 1015.14 examples/s]9942 examples [00:09, 1000.10 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS7GDGJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteS7GDGJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb97dc74950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb97dc74950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb97dc74950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb9e3f01b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb91811aeb8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7fb97dc74950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7fb97dc74950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7fb97dc74950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb97dc61c50>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fb964033128>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7fb8f562d268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7fb8f562d268> 

  function with postional parmater data_info <function split_train_valid at 0x7fb8f562d268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7fb8f562d378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7fb8f562d378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7fb8f562d378> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=061997f92b93fb3a9761d96afbb06beb1c190668d147c5c1b201ec7986fd2703
  Stored in directory: /tmp/pip-ephem-wheel-cache-jgtm3l0v/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
Successfully built en-core-web-sm
Installing collected packages: en-core-web-sm
Successfully installed en-core-web-sm-2.2.5
[38;5;2m✔ Download and installation successful[0m
You can now load the model via spacy.load('en_core_web_sm')
[38;5;2m✔ Linking successful[0m
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/en_core_web_sm
-->
/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/spacy/data/en
You can now load the model via spacy.load('en')
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<21:44:42, 11.0kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<15:27:35, 15.5kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<10:52:36, 22.0kB/s] .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:01<7:37:20, 31.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:01<5:19:19, 44.8kB/s].vector_cache/glove.6B.zip:   1%|          | 9.56M/862M [00:01<3:42:05, 64.0kB/s].vector_cache/glove.6B.zip:   2%|▏         | 14.6M/862M [00:01<2:34:37, 91.4kB/s].vector_cache/glove.6B.zip:   2%|▏         | 18.3M/862M [00:01<1:47:53, 130kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 22.6M/862M [00:01<1:15:14, 186kB/s].vector_cache/glove.6B.zip:   3%|▎         | 26.8M/862M [00:01<52:30, 265kB/s]  .vector_cache/glove.6B.zip:   4%|▎         | 31.4M/862M [00:01<36:38, 378kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.4M/862M [00:02<25:38, 537kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.5M/862M [00:02<17:55, 764kB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.0M/862M [00:02<12:37, 1.08MB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.7M/862M [00:02<08:52, 1.53MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.6M/862M [00:03<07:07, 1.89MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.8M/862M [00:05<06:53, 1.95MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:05<07:01, 1.91MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.0M/862M [00:05<05:30, 2.43MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<06:10, 2.16MB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.3M/862M [00:07<05:46, 2.31MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.8M/862M [00:07<04:22, 3.04MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.0M/862M [00:09<06:09, 2.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.3M/862M [00:09<05:55, 2.24MB/s].vector_cache/glove.6B.zip:   8%|▊         | 66.6M/862M [00:09<04:29, 2.96MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:09<03:19, 3.98MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.2M/862M [00:10<15:29, 853kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:11<12:29, 1.06MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.8M/862M [00:11<09:04, 1.45MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.3M/862M [00:12<09:02, 1.46MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.5M/862M [00:13<09:34, 1.37MB/s].vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:13<07:23, 1.78MB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.6M/862M [00:13<05:26, 2.41MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<07:08, 1.83MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.8M/862M [00:15<06:36, 1.98MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.1M/862M [00:15<04:58, 2.62MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.7M/862M [00:16<06:07, 2.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:17<07:40, 1.69MB/s].vector_cache/glove.6B.zip:  10%|▉         | 82.5M/862M [00:17<06:10, 2.11MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.1M/862M [00:17<04:30, 2.87MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:18<11:22, 1.14MB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:19<09:33, 1.35MB/s].vector_cache/glove.6B.zip:  10%|█         | 87.4M/862M [00:19<07:01, 1.84MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.8M/862M [00:19<05:03, 2.54MB/s].vector_cache/glove.6B.zip:  10%|█         | 90.0M/862M [00:20<37:00, 348kB/s] .vector_cache/glove.6B.zip:  10%|█         | 90.3M/862M [00:21<27:29, 468kB/s].vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:21<19:36, 655kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.1M/862M [00:22<16:18, 785kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:23<12:59, 984kB/s].vector_cache/glove.6B.zip:  11%|█         | 95.7M/862M [00:23<09:25, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.1M/862M [00:23<06:44, 1.89MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:24<40:57, 311kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.6M/862M [00:25<30:16, 420kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.9M/862M [00:25<21:30, 591kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:25<15:09, 836kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:26<3:19:17, 63.5kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<2:21:00, 89.8kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<1:38:56, 128kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<1:11:34, 176kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:28<51:42, 243kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:29<36:31, 344kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<28:01, 447kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<21:09, 592kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:31<15:07, 826kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:31<10:43, 1.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<28:16, 440kB/s] .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<22:51, 545kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<16:45, 742kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<11:52, 1.04MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<16:12, 764kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:34<12:52, 962kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<09:19, 1.33MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:35<06:39, 1.85MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<29:15, 421kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<22:02, 558kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<15:43, 781kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:37<11:07, 1.10MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:38<53:35, 229kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<39:02, 314kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<27:39, 442kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<21:46, 559kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<16:46, 725kB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<12:03, 1.01MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<10:52, 1.11MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:42<09:07, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<06:45, 1.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:44<07:09, 1.68MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<06:32, 1.84MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:44<04:53, 2.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<05:51, 2.04MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:29, 1.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:46<05:55, 2.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<04:22, 2.73MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<06:10, 1.93MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<05:47, 2.05MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:48<04:25, 2.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<05:28, 2.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<06:45, 1.75MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:50<05:29, 2.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:51<04:00, 2.94MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<10:15, 1.15MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<08:40, 1.36MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:52<06:25, 1.83MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:51, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:54<06:14, 1.87MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<04:40, 2.49MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:37, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<06:56, 1.67MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:56<05:34, 2.08MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [00:57<04:04, 2.84MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<10:09, 1.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<08:32, 1.35MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<06:16, 1.84MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<06:43, 1.71MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<06:08, 1.87MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:00<04:38, 2.47MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<05:33, 2.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:02<06:43, 1.70MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<05:24, 2.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:02<03:57, 2.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:03<05:17, 2.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<7:42:27, 24.5kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:04<5:24:07, 35.0kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:04<3:46:11, 49.9kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 185M/862M [01:06<2:44:43, 68.5kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<1:58:20, 95.3kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<1:23:35, 135kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:06<58:28, 192kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<47:48, 234kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<34:51, 321kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:08<24:42, 453kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 194M/862M [01:10<19:29, 572kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:10<15:00, 742kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<10:47, 1.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:46, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<09:36, 1.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<07:26, 1.49MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:12<05:20, 2.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<10:24, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:14<08:43, 1.26MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:14<06:22, 1.72MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:14<04:36, 2.37MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:16<12:43, 859kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:16<10:09, 1.08MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<07:22, 1.48MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:16<05:17, 2.06MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:18<39:50, 273kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:18<29:14, 371kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:18<20:45, 522kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<16:40, 648kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:20<12:54, 836kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:20<09:19, 1.16MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:48, 1.22MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:22<08:58, 1.19MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:22<06:56, 1.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:22<05:00, 2.13MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<10:16, 1.04MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 223M/862M [01:24<08:23, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:24<06:09, 1.72MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<06:35, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:26<05:55, 1.79MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:26<04:25, 2.39MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<05:14, 2.01MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:28<06:19, 1.66MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 232M/862M [01:28<05:06, 2.05MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:28<03:43, 2.80MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:30<09:12, 1.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:30<07:44, 1.35MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:30<05:41, 1.83MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:30<04:06, 2.53MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<19:34, 530kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:32<14:51, 698kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:32<10:40, 970kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<09:41, 1.06MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<09:34, 1.08MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:34<07:14, 1.42MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:34<05:17, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 248M/862M [01:36<06:07, 1.67MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:36<05:33, 1.84MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:36<04:09, 2.46MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:36<03:01, 3.37MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<1:09:36, 146kB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:38<49:59, 203kB/s]  .vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:38<35:11, 288kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<26:35, 380kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:40<19:51, 508kB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:40<14:08, 712kB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:42<11:54, 843kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:42<09:37, 1.04MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:42<07:02, 1.42MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:44<06:55, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<07:18, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:44<05:39, 1.76MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<04:06, 2.41MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:51, 1.69MB/s].vector_cache/glove.6B.zip:  31%|███       | 269M/862M [01:46<05:20, 1.85MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:46<04:02, 2.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:48, 2.04MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:48<04:27, 2.20MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:48<03:23, 2.89MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:29, 2.17MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:50<04:13, 2.31MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:50<03:13, 3.02MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<04:21, 2.22MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:52<05:17, 1.83MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:52<04:14, 2.28MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:52<03:04, 3.13MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:54<12:35, 763kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:54<09:59, 961kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:54<07:15, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:54<05:10, 1.84MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:56<55:19, 173kB/s] .vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:56<39:56, 239kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:56<28:09, 338kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<21:32, 440kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<17:30, 541kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:58<12:45, 742kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:58<09:07, 1.04MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<08:36, 1.09MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:00<07:11, 1.31MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:00<05:19, 1.76MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<05:36, 1.66MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:02<06:13, 1.50MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:02<04:54, 1.90MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:02<03:33, 2.61MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:03<05:33, 1.67MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 306M/862M [02:04<05:02, 1.84MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:04<03:48, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:31, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:06<04:20, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:06<03:18, 2.77MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<04:09, 2.19MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 315M/862M [02:08<05:10, 1.77MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:08<04:05, 2.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:08<03:01, 3.00MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<04:35, 1.98MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:10<04:13, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:10<03:09, 2.85MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:11<04:09, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:12<05:08, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:12<04:06, 2.19MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:12<02:59, 2.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<05:43, 1.56MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:14<05:05, 1.75MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:14<03:45, 2.36MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:14<02:45, 3.21MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:15<10:53, 813kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:16<08:43, 1.01MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:16<06:19, 1.40MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:17<06:11, 1.42MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:18<06:36, 1.33MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:18<05:05, 1.72MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<03:40, 2.37MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<05:53, 1.48MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:19<05:05, 1.71MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:20<03:46, 2.30MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<04:31, 1.91MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<05:07, 1.69MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:22<04:03, 2.12MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:22<02:57, 2.91MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<12:22, 693kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:23<09:42, 882kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:24<07:03, 1.21MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<06:38, 1.28MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<05:42, 1.49MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:26<04:15, 1.99MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:40, 1.80MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:27<04:19, 1.95MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:28<03:14, 2.60MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<03:57, 2.11MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:29<04:51, 1.72MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:30<03:50, 2.17MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<02:49, 2.95MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<04:33, 1.82MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:31<04:03, 2.05MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:31<03:02, 2.71MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:32<02:17, 3.59MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:27, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<06:44, 1.22MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:34<05:13, 1.57MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:34<03:46, 2.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<07:46, 1.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:35<06:26, 1.27MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:36<04:44, 1.71MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<04:57, 1.63MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:37<04:22, 1.85MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:37<03:15, 2.48MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:01, 1.99MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:39<04:49, 1.66MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:39<03:48, 2.10MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:40<02:51, 2.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:41<03:46, 2.11MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<03:32, 2.25MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:41<02:41, 2.94MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:43<03:34, 2.20MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:43<03:23, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:33, 3.07MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:29, 2.24MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<04:32, 1.72MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:45<03:35, 2.17MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<02:38, 2.94MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<04:07, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:47<03:50, 2.01MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:53, 2.66MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<03:35, 2.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:49<04:15, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:49<03:24, 2.24MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:50<02:27, 3.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<09:25, 807kB/s] .vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:51<07:31, 1.01MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:51<05:27, 1.39MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:53<05:20, 1.41MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:53<04:40, 1.61MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:53<03:28, 2.16MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:55<03:56, 1.90MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [02:55<03:41, 2.02MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:55<02:46, 2.67MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:56<02:22, 3.12MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 418M/862M [02:57<5:13:41, 23.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:57<3:39:36, 33.6kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [02:57<2:33:23, 48.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:59<1:48:45, 67.4kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<1:17:55, 94.0kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [02:59<54:55, 133kB/s]   .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [02:59<38:19, 190kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 427M/862M [03:01<31:24, 231kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:01<22:51, 317kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:01<16:10, 447kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<12:43, 565kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:03<09:47, 734kB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:03<07:03, 1.01MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<06:21, 1.12MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:05<05:19, 1.33MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:05<03:56, 1.80MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:10, 1.69MB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:07<04:39, 1.51MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:07<03:38, 1.93MB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:07<02:40, 2.62MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:42, 1.89MB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:09<03:27, 2.02MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:09<02:37, 2.65MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:11<03:13, 2.14MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:11<03:06, 2.22MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:11<02:21, 2.92MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:13<03:02, 2.25MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:13<02:52, 2.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:13<02:09, 3.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:13<01:35, 4.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<6:48:09, 16.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:15<4:46:18, 23.6kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:15<3:19:57, 33.7kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:15<2:19:10, 48.2kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<3:51:13, 29.0kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:17<2:42:35, 41.2kB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 461M/862M [03:17<1:53:37, 58.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<1:20:23, 82.6kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:19<57:05, 116kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:19<40:00, 165kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:19<27:54, 235kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<1:15:39, 86.8kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:21<54:33, 120kB/s]   .vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:21<38:32, 170kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:21<26:54, 242kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:23<22:53, 284kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:23<16:49, 386kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<11:56, 542kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:25<09:36, 669kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:25<07:31, 855kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<05:26, 1.18MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<05:04, 1.25MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:27<04:21, 1.46MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<03:14, 1.95MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<03:31, 1.78MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:29<04:01, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:29<03:12, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:29<02:18, 2.69MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<05:22, 1.16MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:31<04:33, 1.36MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:22, 1.84MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:35, 1.71MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:33<03:16, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<02:28, 2.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<02:57, 2.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:35<03:39, 1.66MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:35<02:55, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:35<02:07, 2.84MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:37<05:12, 1.15MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:37<04:23, 1.37MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:13, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:26, 1.72MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:57, 1.50MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:39<03:05, 1.92MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<02:14, 2.63MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:28, 1.69MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:41<03:09, 1.86MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:41<02:21, 2.48MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<02:49, 2.06MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:43<03:25, 1.69MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:43<02:45, 2.10MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:43<02:00, 2.86MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:20, 1.72MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:45<03:04, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<02:17, 2.49MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<02:44, 2.06MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:47<03:19, 1.70MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:47<02:37, 2.15MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<01:58, 2.85MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:48<02:38, 2.13MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:49<02:32, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<01:57, 2.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:50<02:28, 2.23MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:51<02:25, 2.28MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<01:51, 2.95MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:52<02:24, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<03:01, 1.80MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:53<02:24, 2.27MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<01:48, 3.01MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 539M/862M [03:54<02:31, 2.14MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:55<02:18, 2.33MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:55<01:47, 2.99MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [03:55<01:19, 4.01MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:56<03:52, 1.37MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [03:57<04:05, 1.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:57<03:08, 1.69MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<02:17, 2.30MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:58<02:56, 1.79MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [03:59<02:41, 1.95MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<02:00, 2.60MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:00<02:27, 2.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<03:00, 1.73MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:01<02:25, 2.13MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<01:45, 2.92MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:02<04:24, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:03<03:42, 1.38MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:43, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:04<02:55, 1.73MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:05<03:17, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 560M/862M [04:05<02:34, 1.96MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:05<01:51, 2.69MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<03:14, 1.53MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:06<02:53, 1.71MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<02:09, 2.29MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:29, 1.96MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:08<02:18, 2.13MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:09<01:42, 2.84MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:15, 2.14MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:10<02:10, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<01:39, 2.88MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:07, 2.24MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:12<02:33, 1.86MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<02:03, 2.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:13<01:29, 3.15MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:14<06:42, 700kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:14<05:16, 889kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 582M/862M [04:15<03:48, 1.23MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:35, 1.29MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:16<03:39, 1.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:17<02:47, 1.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:17<02:00, 2.28MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<03:04, 1.48MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:18<02:42, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:00, 2.25MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:27, 3.09MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<15:59, 281kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:20<12:17, 365kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<08:51, 505kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<06:12, 713kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<07:08, 618kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:22<05:32, 796kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<03:58, 1.10MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:23<02:49, 1.54MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 601M/862M [04:24<07:45, 561kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:24<05:54, 735kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:24<04:12, 1.02MB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:26<03:51, 1.11MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<03:40, 1.16MB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:26<02:46, 1.54MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<01:59, 2.13MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:56, 1.43MB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:28<02:31, 1.66MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:28<01:51, 2.25MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:11, 1.89MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:30<02:33, 1.62MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:02, 2.02MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:28, 2.76MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:35, 1.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:32<03:02, 1.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:32<02:13, 1.82MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:21, 1.70MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:34<02:40, 1.50MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<02:06, 1.89MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:35<01:31, 2.60MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:36<03:28, 1.13MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:55, 1.34MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:36<02:09, 1.80MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:38<02:16, 1.69MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:33, 1.51MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<02:01, 1.90MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:39<01:27, 2.62MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<03:22, 1.12MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:40<02:50, 1.33MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:40<02:04, 1.81MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:11, 1.69MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:42<02:27, 1.51MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<01:54, 1.94MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:42<01:23, 2.63MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:57, 1.86MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:44<01:50, 1.98MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:44<01:22, 2.63MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:41, 2.13MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:46<01:37, 2.21MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:46<01:14, 2.87MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:47<01:04, 3.29MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:48<2:28:04, 23.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<1:43:32, 33.9kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:48<1:11:38, 48.4kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<51:52, 66.5kB/s]  .vector_cache/glove.6B.zip:  76%|███████▌  | 655M/862M [04:50<37:08, 92.8kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<26:07, 132kB/s] .vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:50<18:07, 187kB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:52<14:34, 232kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<10:36, 318kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<07:28, 449kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [04:54<05:50, 567kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<04:54, 674kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:54<03:36, 916kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:54<02:32, 1.28MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:55, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [04:56<02:26, 1.32MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<01:48, 1.78MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<01:53, 1.68MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [04:58<02:06, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<01:38, 1.93MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:58<01:11, 2.63MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:42, 1.82MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:00<01:33, 1.99MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<01:10, 2.63MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:27, 2.07MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:02<01:21, 2.23MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<01:01, 2.93MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:21, 2.18MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:04<01:41, 1.75MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:04<01:20, 2.21MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<00:58, 3.00MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:06<01:37, 1.79MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:06<01:29, 1.94MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:07, 2.55MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:21, 2.09MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:38, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:08<01:18, 2.15MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<00:56, 2.92MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<02:16, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:10<01:55, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:25, 1.92MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:31, 1.76MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:12<01:45, 1.52MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:22, 1.94MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:12<00:58, 2.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<02:06, 1.24MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:14<01:47, 1.45MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<01:19, 1.95MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:16<01:25, 1.78MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<01:19, 1.92MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<00:59, 2.52MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:18<01:11, 2.08MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<01:08, 2.17MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<00:51, 2.85MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:04, 2.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:18, 1.84MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:20<01:01, 2.34MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<00:44, 3.17MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:18, 1.80MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 722M/862M [05:22<01:12, 1.94MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<00:54, 2.54MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:05, 2.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 726M/862M [05:24<01:02, 2.18MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:24<00:46, 2.89MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:24<00:33, 3.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<03:23, 648kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 730M/862M [05:26<02:55, 750kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<02:10, 1.00MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<01:31, 1.40MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 734M/862M [05:28<02:24, 888kB/s] .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<01:54, 1.11MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<01:22, 1.53MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:30<01:23, 1.48MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:29, 1.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<01:09, 1.76MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:30<00:49, 2.42MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:48, 1.10MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<01:31, 1.31MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<01:06, 1.78MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:32<00:46, 2.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<16:58, 113kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<12:05, 159kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<08:25, 225kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<06:08, 302kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<04:31, 409kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<03:10, 576kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:32, 705kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:38<02:13, 803kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:39, 1.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:38<01:09, 1.49MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 759M/862M [05:40<01:53, 908kB/s] .vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<01:31, 1.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<01:06, 1.53MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:40<00:46, 2.12MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:42<03:55, 420kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:54, 564kB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<02:05, 779kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:42<01:27, 1.10MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:43<01:53, 838kB/s] .vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<01:30, 1.04MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<01:05, 1.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:45<01:02, 1.44MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<01:06, 1.36MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<00:51, 1.76MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<00:37, 2.37MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:47<00:45, 1.89MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<00:42, 2.02MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:31, 2.65MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:49<00:38, 2.14MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 780M/862M [05:50<00:47, 1.74MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:37, 2.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:50<00:26, 2.93MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:51<01:08, 1.15MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 784M/862M [05:52<00:57, 1.36MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:41, 1.85MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:53<00:42, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:54<00:49, 1.50MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<00:38, 1.88MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:27, 2.58MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:55<01:02, 1.11MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<00:52, 1.32MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [05:56<00:38, 1.78MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:39, 1.68MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:44, 1.48MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:34, 1.87MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 800M/862M [05:58<00:24, 2.56MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<00:55, 1.10MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:00<00:46, 1.32MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<00:33, 1.78MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:34, 1.67MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:30, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:02<00:22, 2.46MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:15, 3.35MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:15, 700kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<00:59, 889kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:41, 1.23MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:05<00:37, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:32, 1.49MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:06<00:23, 1.99MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:07<00:24, 1.81MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:22, 1.94MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:08<00:16, 2.58MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:19, 2.11MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:10<00:17, 2.23MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:10<00:12, 3.07MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:47, 771kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:37, 969kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:12<00:26, 1.33MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:23, 1.37MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<00:20, 1.57MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:13<00:14, 2.12MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:09, 2.92MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<28:09, 16.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<19:42, 23.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<13:30, 33.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<09:03, 48.1kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:17<05:54, 67.7kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<04:06, 95.9kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:17<02:43, 136kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<01:41, 194kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:19<21:02, 15.7kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<14:41, 22.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<09:57, 31.8kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:20<06:03, 45.4kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<04:10, 62.5kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<02:53, 88.3kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<01:51, 126kB/s] .vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<01:05, 179kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<01:11, 161kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:49, 224kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:23<00:31, 316kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:17, 414kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:12, 548kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:07, 765kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:03, 895kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:02, 1.10MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:01, 1.50MB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<29:58:14,  3.71it/s]  0%|          | 899/400000 [00:00<20:56:10,  5.30it/s]  0%|          | 1794/400000 [00:00<14:37:34,  7.56it/s]  1%|          | 2702/400000 [00:00<10:13:07, 10.80it/s]  1%|          | 3578/400000 [00:00<7:08:27, 15.42it/s]   1%|          | 4445/400000 [00:00<4:59:29, 22.01it/s]  1%|▏         | 5332/400000 [00:00<3:29:23, 31.41it/s]  2%|▏         | 6190/400000 [00:00<2:26:29, 44.80it/s]  2%|▏         | 7082/400000 [00:01<1:42:31, 63.87it/s]  2%|▏         | 7960/400000 [00:01<1:11:50, 90.96it/s]  2%|▏         | 8842/400000 [00:01<50:23, 129.37it/s]   2%|▏         | 9751/400000 [00:01<35:24, 183.69it/s]  3%|▎         | 10645/400000 [00:01<24:56, 260.13it/s]  3%|▎         | 11548/400000 [00:01<17:38, 367.07it/s]  3%|▎         | 12434/400000 [00:01<12:32, 515.03it/s]  3%|▎         | 13321/400000 [00:01<08:58, 717.87it/s]  4%|▎         | 14203/400000 [00:01<06:30, 987.47it/s]  4%|▍         | 15096/400000 [00:01<04:45, 1346.80it/s]  4%|▍         | 15965/400000 [00:02<03:32, 1804.16it/s]  4%|▍         | 16833/400000 [00:02<02:42, 2356.92it/s]  4%|▍         | 17689/400000 [00:02<02:07, 2994.03it/s]  5%|▍         | 18551/400000 [00:02<01:42, 3722.77it/s]  5%|▍         | 19405/400000 [00:02<01:24, 4480.49it/s]  5%|▌         | 20255/400000 [00:02<01:12, 5212.03it/s]  5%|▌         | 21103/400000 [00:02<01:04, 5847.36it/s]  5%|▌         | 21951/400000 [00:02<00:58, 6447.09it/s]  6%|▌         | 22793/400000 [00:02<00:54, 6931.09it/s]  6%|▌         | 23675/400000 [00:02<00:50, 7406.87it/s]  6%|▌         | 24585/400000 [00:03<00:47, 7844.49it/s]  6%|▋         | 25458/400000 [00:03<00:46, 8089.54it/s]  7%|▋         | 26330/400000 [00:03<00:45, 8266.81it/s]  7%|▋         | 27220/400000 [00:03<00:44, 8446.79it/s]  7%|▋         | 28097/400000 [00:03<00:43, 8539.84it/s]  7%|▋         | 28993/400000 [00:03<00:42, 8660.17it/s]  7%|▋         | 29892/400000 [00:03<00:42, 8755.74it/s]  8%|▊         | 30780/400000 [00:03<00:42, 8688.04it/s]  8%|▊         | 31658/400000 [00:03<00:42, 8677.97it/s]  8%|▊         | 32532/400000 [00:04<00:42, 8650.48it/s]  8%|▊         | 33411/400000 [00:04<00:42, 8690.74it/s]  9%|▊         | 34283/400000 [00:04<00:42, 8643.88it/s]  9%|▉         | 35151/400000 [00:04<00:42, 8654.03it/s]  9%|▉         | 36018/400000 [00:04<00:42, 8645.12it/s]  9%|▉         | 36899/400000 [00:04<00:41, 8693.56it/s]  9%|▉         | 37774/400000 [00:04<00:41, 8707.79it/s] 10%|▉         | 38652/400000 [00:04<00:41, 8728.27it/s] 10%|▉         | 39526/400000 [00:04<00:41, 8720.38it/s] 10%|█         | 40399/400000 [00:04<00:42, 8518.66it/s] 10%|█         | 41253/400000 [00:05<00:43, 8262.33it/s] 11%|█         | 42128/400000 [00:05<00:42, 8400.77it/s] 11%|█         | 43010/400000 [00:05<00:41, 8520.27it/s] 11%|█         | 43895/400000 [00:05<00:41, 8613.91it/s] 11%|█         | 44776/400000 [00:05<00:40, 8670.43it/s] 11%|█▏        | 45676/400000 [00:05<00:40, 8764.94it/s] 12%|█▏        | 46554/400000 [00:05<00:40, 8768.36it/s] 12%|█▏        | 47432/400000 [00:05<00:40, 8769.75it/s] 12%|█▏        | 48319/400000 [00:05<00:39, 8794.01it/s] 12%|█▏        | 49200/400000 [00:05<00:39, 8798.21it/s] 13%|█▎        | 50081/400000 [00:06<00:40, 8658.98it/s] 13%|█▎        | 50948/400000 [00:06<00:40, 8631.64it/s] 13%|█▎        | 51816/400000 [00:06<00:40, 8643.60it/s] 13%|█▎        | 52681/400000 [00:06<00:40, 8618.42it/s] 13%|█▎        | 53580/400000 [00:06<00:39, 8725.49it/s] 14%|█▎        | 54454/400000 [00:06<00:39, 8722.55it/s] 14%|█▍        | 55357/400000 [00:06<00:39, 8811.00it/s] 14%|█▍        | 56279/400000 [00:06<00:38, 8928.66it/s] 14%|█▍        | 57205/400000 [00:06<00:37, 9025.00it/s] 15%|█▍        | 58109/400000 [00:06<00:38, 8888.69it/s] 15%|█▍        | 59006/400000 [00:07<00:38, 8911.45it/s] 15%|█▍        | 59898/400000 [00:07<00:38, 8872.70it/s] 15%|█▌        | 60786/400000 [00:07<00:38, 8716.23it/s] 15%|█▌        | 61659/400000 [00:07<00:39, 8541.45it/s] 16%|█▌        | 62557/400000 [00:07<00:38, 8665.58it/s] 16%|█▌        | 63465/400000 [00:07<00:38, 8783.12it/s] 16%|█▌        | 64364/400000 [00:07<00:37, 8843.30it/s] 16%|█▋        | 65272/400000 [00:07<00:37, 8910.82it/s] 17%|█▋        | 66164/400000 [00:07<00:38, 8779.10it/s] 17%|█▋        | 67043/400000 [00:07<00:39, 8518.56it/s] 17%|█▋        | 67930/400000 [00:08<00:38, 8619.91it/s] 17%|█▋        | 68800/400000 [00:08<00:38, 8641.76it/s] 17%|█▋        | 69691/400000 [00:08<00:37, 8720.04it/s] 18%|█▊        | 70592/400000 [00:08<00:37, 8803.71it/s] 18%|█▊        | 71474/400000 [00:08<00:37, 8805.62it/s] 18%|█▊        | 72388/400000 [00:08<00:36, 8902.90it/s] 18%|█▊        | 73280/400000 [00:08<00:37, 8793.00it/s] 19%|█▊        | 74170/400000 [00:08<00:36, 8823.74it/s] 19%|█▉        | 75053/400000 [00:08<00:37, 8617.46it/s] 19%|█▉        | 75917/400000 [00:08<00:38, 8466.29it/s] 19%|█▉        | 76811/400000 [00:09<00:37, 8600.56it/s] 19%|█▉        | 77699/400000 [00:09<00:37, 8681.26it/s] 20%|█▉        | 78598/400000 [00:09<00:36, 8755.87it/s] 20%|█▉        | 79475/400000 [00:09<00:36, 8683.23it/s] 20%|██        | 80368/400000 [00:09<00:36, 8753.19it/s] 20%|██        | 81266/400000 [00:09<00:36, 8818.72it/s] 21%|██        | 82149/400000 [00:09<00:36, 8817.07it/s] 21%|██        | 83032/400000 [00:09<00:36, 8618.72it/s] 21%|██        | 83903/400000 [00:09<00:36, 8643.23it/s] 21%|██        | 84769/400000 [00:10<00:36, 8645.42it/s] 21%|██▏       | 85656/400000 [00:10<00:36, 8710.29it/s] 22%|██▏       | 86528/400000 [00:10<00:36, 8683.34it/s] 22%|██▏       | 87397/400000 [00:10<00:36, 8511.52it/s] 22%|██▏       | 88253/400000 [00:10<00:36, 8524.69it/s] 22%|██▏       | 89158/400000 [00:10<00:35, 8674.37it/s] 23%|██▎       | 90055/400000 [00:10<00:35, 8759.73it/s] 23%|██▎       | 90932/400000 [00:10<00:35, 8746.28it/s] 23%|██▎       | 91823/400000 [00:10<00:35, 8794.66it/s] 23%|██▎       | 92704/400000 [00:10<00:36, 8433.52it/s] 23%|██▎       | 93633/400000 [00:11<00:35, 8671.68it/s] 24%|██▎       | 94505/400000 [00:11<00:35, 8684.36it/s] 24%|██▍       | 95377/400000 [00:11<00:35, 8664.24it/s] 24%|██▍       | 96246/400000 [00:11<00:35, 8616.02it/s] 24%|██▍       | 97117/400000 [00:11<00:35, 8643.09it/s] 25%|██▍       | 98004/400000 [00:11<00:34, 8709.66it/s] 25%|██▍       | 98924/400000 [00:11<00:34, 8850.12it/s] 25%|██▍       | 99811/400000 [00:11<00:34, 8823.74it/s] 25%|██▌       | 100698/400000 [00:11<00:33, 8835.89it/s] 25%|██▌       | 101625/400000 [00:11<00:33, 8959.74it/s] 26%|██▌       | 102526/400000 [00:12<00:33, 8972.01it/s] 26%|██▌       | 103484/400000 [00:12<00:32, 9145.53it/s] 26%|██▌       | 104415/400000 [00:12<00:32, 9193.42it/s] 26%|██▋       | 105336/400000 [00:12<00:32, 9083.98it/s] 27%|██▋       | 106246/400000 [00:12<00:32, 9001.50it/s] 27%|██▋       | 107147/400000 [00:12<00:32, 8902.13it/s] 27%|██▋       | 108057/400000 [00:12<00:32, 8957.88it/s] 27%|██▋       | 108957/400000 [00:12<00:32, 8969.53it/s] 27%|██▋       | 109889/400000 [00:12<00:31, 9069.88it/s] 28%|██▊       | 110797/400000 [00:12<00:32, 8903.01it/s] 28%|██▊       | 111689/400000 [00:13<00:33, 8688.22it/s] 28%|██▊       | 112560/400000 [00:13<00:33, 8649.99it/s] 28%|██▊       | 113446/400000 [00:13<00:32, 8711.10it/s] 29%|██▊       | 114352/400000 [00:13<00:32, 8812.76it/s] 29%|██▉       | 115235/400000 [00:13<00:32, 8769.93it/s] 29%|██▉       | 116113/400000 [00:13<00:32, 8740.73it/s] 29%|██▉       | 116988/400000 [00:13<00:33, 8568.18it/s] 29%|██▉       | 117873/400000 [00:13<00:32, 8650.21it/s] 30%|██▉       | 118750/400000 [00:13<00:32, 8683.60it/s] 30%|██▉       | 119620/400000 [00:13<00:32, 8634.54it/s] 30%|███       | 120493/400000 [00:14<00:32, 8661.88it/s] 30%|███       | 121380/400000 [00:14<00:31, 8723.19it/s] 31%|███       | 122298/400000 [00:14<00:31, 8854.87it/s] 31%|███       | 123194/400000 [00:14<00:31, 8885.28it/s] 31%|███       | 124084/400000 [00:14<00:31, 8683.29it/s] 31%|███       | 124972/400000 [00:14<00:31, 8740.75it/s] 31%|███▏      | 125886/400000 [00:14<00:30, 8856.46it/s] 32%|███▏      | 126773/400000 [00:14<00:31, 8742.57it/s] 32%|███▏      | 127649/400000 [00:14<00:31, 8608.98it/s] 32%|███▏      | 128512/400000 [00:14<00:31, 8512.67it/s] 32%|███▏      | 129371/400000 [00:15<00:31, 8529.85it/s] 33%|███▎      | 130225/400000 [00:15<00:32, 8405.30it/s] 33%|███▎      | 131124/400000 [00:15<00:31, 8571.01it/s] 33%|███▎      | 132029/400000 [00:15<00:30, 8708.41it/s] 33%|███▎      | 132902/400000 [00:15<00:30, 8679.20it/s] 33%|███▎      | 133775/400000 [00:15<00:30, 8694.33it/s] 34%|███▎      | 134662/400000 [00:15<00:30, 8745.05it/s] 34%|███▍      | 135538/400000 [00:15<00:30, 8735.90it/s] 34%|███▍      | 136417/400000 [00:15<00:30, 8750.09it/s] 34%|███▍      | 137293/400000 [00:16<00:30, 8572.90it/s] 35%|███▍      | 138152/400000 [00:16<00:30, 8462.53it/s] 35%|███▍      | 139000/400000 [00:16<00:31, 8272.25it/s] 35%|███▍      | 139835/400000 [00:16<00:31, 8293.74it/s] 35%|███▌      | 140666/400000 [00:16<00:32, 8005.90it/s] 35%|███▌      | 141470/400000 [00:16<00:32, 7944.70it/s] 36%|███▌      | 142284/400000 [00:16<00:32, 7998.09it/s] 36%|███▌      | 143086/400000 [00:16<00:32, 7953.68it/s] 36%|███▌      | 143905/400000 [00:16<00:31, 8021.41it/s] 36%|███▌      | 144746/400000 [00:16<00:31, 8134.08it/s] 36%|███▋      | 145561/400000 [00:17<00:32, 7827.60it/s] 37%|███▋      | 146455/400000 [00:17<00:31, 8129.24it/s] 37%|███▋      | 147330/400000 [00:17<00:30, 8305.27it/s] 37%|███▋      | 148310/400000 [00:17<00:28, 8701.34it/s] 37%|███▋      | 149229/400000 [00:17<00:28, 8840.02it/s] 38%|███▊      | 150120/400000 [00:17<00:28, 8826.16it/s] 38%|███▊      | 151009/400000 [00:17<00:28, 8844.30it/s] 38%|███▊      | 151901/400000 [00:17<00:27, 8865.32it/s] 38%|███▊      | 152790/400000 [00:17<00:28, 8763.13it/s] 38%|███▊      | 153669/400000 [00:17<00:28, 8691.56it/s] 39%|███▊      | 154540/400000 [00:18<00:28, 8696.18it/s] 39%|███▉      | 155435/400000 [00:18<00:27, 8768.31it/s] 39%|███▉      | 156327/400000 [00:18<00:27, 8809.92it/s] 39%|███▉      | 157209/400000 [00:18<00:27, 8803.94it/s] 40%|███▉      | 158117/400000 [00:18<00:27, 8883.32it/s] 40%|███▉      | 159006/400000 [00:18<00:27, 8782.83it/s] 40%|███▉      | 159895/400000 [00:18<00:27, 8813.16it/s] 40%|████      | 160819/400000 [00:18<00:26, 8935.54it/s] 40%|████      | 161714/400000 [00:18<00:26, 8829.43it/s] 41%|████      | 162620/400000 [00:18<00:26, 8897.24it/s] 41%|████      | 163535/400000 [00:19<00:26, 8970.14it/s] 41%|████      | 164514/400000 [00:19<00:25, 9199.32it/s] 41%|████▏     | 165436/400000 [00:19<00:26, 9020.85it/s] 42%|████▏     | 166345/400000 [00:19<00:25, 9039.16it/s] 42%|████▏     | 167251/400000 [00:19<00:26, 8868.26it/s] 42%|████▏     | 168140/400000 [00:19<00:26, 8812.73it/s] 42%|████▏     | 169023/400000 [00:19<00:26, 8645.26it/s] 42%|████▏     | 169925/400000 [00:19<00:26, 8754.33it/s] 43%|████▎     | 170802/400000 [00:19<00:26, 8734.94it/s] 43%|████▎     | 171677/400000 [00:20<00:27, 8262.75it/s] 43%|████▎     | 172510/400000 [00:20<00:27, 8281.71it/s] 43%|████▎     | 173343/400000 [00:20<00:27, 8291.57it/s] 44%|████▎     | 174268/400000 [00:20<00:26, 8556.43it/s] 44%|████▍     | 175161/400000 [00:20<00:25, 8662.98it/s] 44%|████▍     | 176073/400000 [00:20<00:25, 8793.49it/s] 44%|████▍     | 176971/400000 [00:20<00:25, 8846.66it/s] 44%|████▍     | 177878/400000 [00:20<00:24, 8909.81it/s] 45%|████▍     | 178771/400000 [00:20<00:25, 8763.73it/s] 45%|████▍     | 179649/400000 [00:20<00:25, 8766.25it/s] 45%|████▌     | 180527/400000 [00:21<00:25, 8712.91it/s] 45%|████▌     | 181400/400000 [00:21<00:25, 8588.11it/s] 46%|████▌     | 182260/400000 [00:21<00:25, 8531.22it/s] 46%|████▌     | 183129/400000 [00:21<00:25, 8523.24it/s] 46%|████▌     | 184106/400000 [00:21<00:24, 8861.41it/s] 46%|████▋     | 185026/400000 [00:21<00:23, 8960.29it/s] 46%|████▋     | 185925/400000 [00:21<00:23, 8968.18it/s] 47%|████▋     | 186897/400000 [00:21<00:23, 9180.84it/s] 47%|████▋     | 187818/400000 [00:21<00:23, 9045.04it/s] 47%|████▋     | 188725/400000 [00:21<00:24, 8702.59it/s] 47%|████▋     | 189600/400000 [00:22<00:24, 8682.89it/s] 48%|████▊     | 190486/400000 [00:22<00:23, 8733.96it/s] 48%|████▊     | 191362/400000 [00:22<00:23, 8725.56it/s] 48%|████▊     | 192304/400000 [00:22<00:23, 8922.12it/s] 48%|████▊     | 193224/400000 [00:22<00:22, 9002.85it/s] 49%|████▊     | 194126/400000 [00:22<00:23, 8917.47it/s] 49%|████▉     | 195020/400000 [00:22<00:23, 8839.64it/s] 49%|████▉     | 195952/400000 [00:22<00:22, 8976.88it/s] 49%|████▉     | 196851/400000 [00:22<00:22, 8860.63it/s] 49%|████▉     | 197739/400000 [00:22<00:23, 8612.46it/s] 50%|████▉     | 198632/400000 [00:23<00:23, 8704.52it/s] 50%|████▉     | 199520/400000 [00:23<00:22, 8755.31it/s] 50%|█████     | 200403/400000 [00:23<00:22, 8777.19it/s] 50%|█████     | 201322/400000 [00:23<00:22, 8894.93it/s] 51%|█████     | 202253/400000 [00:23<00:21, 9015.40it/s] 51%|█████     | 203165/400000 [00:23<00:21, 9043.21it/s] 51%|█████     | 204071/400000 [00:23<00:22, 8867.28it/s] 51%|█████     | 204983/400000 [00:23<00:21, 8939.85it/s] 51%|█████▏    | 205899/400000 [00:23<00:21, 9003.58it/s] 52%|█████▏    | 206801/400000 [00:23<00:21, 8843.75it/s] 52%|█████▏    | 207687/400000 [00:24<00:21, 8843.67it/s] 52%|█████▏    | 208573/400000 [00:24<00:21, 8828.04it/s] 52%|█████▏    | 209458/400000 [00:24<00:21, 8833.12it/s] 53%|█████▎    | 210345/400000 [00:24<00:21, 8842.50it/s] 53%|█████▎    | 211230/400000 [00:24<00:21, 8782.44it/s] 53%|█████▎    | 212116/400000 [00:24<00:21, 8803.22it/s] 53%|█████▎    | 213005/400000 [00:24<00:21, 8828.45it/s] 53%|█████▎    | 213922/400000 [00:24<00:20, 8925.88it/s] 54%|█████▎    | 214815/400000 [00:24<00:21, 8608.18it/s] 54%|█████▍    | 215679/400000 [00:25<00:22, 8365.29it/s] 54%|█████▍    | 216587/400000 [00:25<00:21, 8566.02it/s] 54%|█████▍    | 217448/400000 [00:25<00:21, 8550.37it/s] 55%|█████▍    | 218349/400000 [00:25<00:20, 8681.44it/s] 55%|█████▍    | 219242/400000 [00:25<00:20, 8751.98it/s] 55%|█████▌    | 220142/400000 [00:25<00:20, 8822.60it/s] 55%|█████▌    | 221035/400000 [00:25<00:20, 8854.30it/s] 55%|█████▌    | 221922/400000 [00:25<00:20, 8856.41it/s] 56%|█████▌    | 222809/400000 [00:25<00:20, 8851.08it/s] 56%|█████▌    | 223695/400000 [00:25<00:20, 8772.83it/s] 56%|█████▌    | 224573/400000 [00:26<00:20, 8736.34it/s] 56%|█████▋    | 225448/400000 [00:26<00:20, 8593.30it/s] 57%|█████▋    | 226309/400000 [00:26<00:20, 8555.42it/s] 57%|█████▋    | 227201/400000 [00:26<00:19, 8661.60it/s] 57%|█████▋    | 228068/400000 [00:26<00:20, 8578.32it/s] 57%|█████▋    | 228966/400000 [00:26<00:19, 8694.02it/s] 57%|█████▋    | 229842/400000 [00:26<00:19, 8713.18it/s] 58%|█████▊    | 230714/400000 [00:26<00:19, 8550.60it/s] 58%|█████▊    | 231571/400000 [00:26<00:19, 8522.33it/s] 58%|█████▊    | 232425/400000 [00:26<00:20, 8322.18it/s] 58%|█████▊    | 233382/400000 [00:27<00:19, 8658.65it/s] 59%|█████▊    | 234300/400000 [00:27<00:18, 8808.25it/s] 59%|█████▉    | 235185/400000 [00:27<00:19, 8542.76it/s] 59%|█████▉    | 236065/400000 [00:27<00:19, 8617.71it/s] 59%|█████▉    | 236935/400000 [00:27<00:18, 8640.66it/s] 59%|█████▉    | 237836/400000 [00:27<00:18, 8745.90it/s] 60%|█████▉    | 238713/400000 [00:27<00:18, 8632.43it/s] 60%|█████▉    | 239595/400000 [00:27<00:18, 8686.49it/s] 60%|██████    | 240491/400000 [00:27<00:18, 8765.93it/s] 60%|██████    | 241369/400000 [00:27<00:18, 8525.49it/s] 61%|██████    | 242269/400000 [00:28<00:18, 8660.43it/s] 61%|██████    | 243182/400000 [00:28<00:17, 8793.79it/s] 61%|██████    | 244088/400000 [00:28<00:17, 8870.79it/s] 61%|██████    | 244977/400000 [00:28<00:17, 8743.55it/s] 61%|██████▏   | 245853/400000 [00:28<00:17, 8669.54it/s] 62%|██████▏   | 246775/400000 [00:28<00:17, 8827.49it/s] 62%|██████▏   | 247688/400000 [00:28<00:17, 8915.91it/s] 62%|██████▏   | 248581/400000 [00:28<00:17, 8533.63it/s] 62%|██████▏   | 249457/400000 [00:28<00:17, 8599.37it/s] 63%|██████▎   | 250321/400000 [00:29<00:17, 8465.03it/s] 63%|██████▎   | 251220/400000 [00:29<00:17, 8614.59it/s] 63%|██████▎   | 252136/400000 [00:29<00:16, 8770.82it/s] 63%|██████▎   | 253047/400000 [00:29<00:16, 8869.30it/s] 63%|██████▎   | 253936/400000 [00:29<00:16, 8762.45it/s] 64%|██████▎   | 254814/400000 [00:29<00:16, 8683.07it/s] 64%|██████▍   | 255687/400000 [00:29<00:16, 8695.79it/s] 64%|██████▍   | 256594/400000 [00:29<00:16, 8803.81it/s] 64%|██████▍   | 257476/400000 [00:29<00:16, 8701.35it/s] 65%|██████▍   | 258364/400000 [00:29<00:16, 8752.32it/s] 65%|██████▍   | 259282/400000 [00:30<00:15, 8875.16it/s] 65%|██████▌   | 260171/400000 [00:30<00:16, 8551.29it/s] 65%|██████▌   | 261030/400000 [00:30<00:16, 8447.98it/s] 65%|██████▌   | 261878/400000 [00:30<00:16, 8425.59it/s] 66%|██████▌   | 262771/400000 [00:30<00:16, 8569.16it/s] 66%|██████▌   | 263676/400000 [00:30<00:15, 8706.92it/s] 66%|██████▌   | 264567/400000 [00:30<00:15, 8766.28it/s] 66%|██████▋   | 265446/400000 [00:30<00:15, 8746.79it/s] 67%|██████▋   | 266322/400000 [00:30<00:15, 8391.50it/s] 67%|██████▋   | 267236/400000 [00:30<00:15, 8601.08it/s] 67%|██████▋   | 268101/400000 [00:31<00:15, 8607.58it/s] 67%|██████▋   | 268998/400000 [00:31<00:15, 8710.99it/s] 67%|██████▋   | 269872/400000 [00:31<00:15, 8628.05it/s] 68%|██████▊   | 270737/400000 [00:31<00:15, 8540.70it/s] 68%|██████▊   | 271634/400000 [00:31<00:14, 8663.37it/s] 68%|██████▊   | 272502/400000 [00:31<00:14, 8620.99it/s] 68%|██████▊   | 273388/400000 [00:31<00:14, 8691.04it/s] 69%|██████▊   | 274258/400000 [00:31<00:14, 8645.75it/s] 69%|██████▉   | 275124/400000 [00:31<00:15, 8102.71it/s] 69%|██████▉   | 275979/400000 [00:31<00:15, 8230.01it/s] 69%|██████▉   | 276829/400000 [00:32<00:14, 8308.64it/s] 69%|██████▉   | 277692/400000 [00:32<00:14, 8400.05it/s] 70%|██████▉   | 278552/400000 [00:32<00:14, 8457.91it/s] 70%|██████▉   | 279427/400000 [00:32<00:14, 8541.27it/s] 70%|███████   | 280283/400000 [00:32<00:14, 8335.55it/s] 70%|███████   | 281144/400000 [00:32<00:14, 8414.99it/s] 71%|███████   | 282010/400000 [00:32<00:13, 8486.60it/s] 71%|███████   | 282861/400000 [00:32<00:13, 8491.09it/s] 71%|███████   | 283712/400000 [00:32<00:13, 8466.81it/s] 71%|███████   | 284613/400000 [00:32<00:13, 8622.29it/s] 71%|███████▏  | 285477/400000 [00:33<00:13, 8405.59it/s] 72%|███████▏  | 286376/400000 [00:33<00:13, 8571.02it/s] 72%|███████▏  | 287285/400000 [00:33<00:12, 8719.85it/s] 72%|███████▏  | 288167/400000 [00:33<00:12, 8746.27it/s] 72%|███████▏  | 289044/400000 [00:33<00:12, 8614.30it/s] 72%|███████▏  | 289907/400000 [00:33<00:12, 8589.95it/s] 73%|███████▎  | 290768/400000 [00:33<00:13, 8167.60it/s] 73%|███████▎  | 291647/400000 [00:33<00:12, 8343.60it/s] 73%|███████▎  | 292529/400000 [00:33<00:12, 8479.28it/s] 73%|███████▎  | 293390/400000 [00:34<00:12, 8516.66it/s] 74%|███████▎  | 294276/400000 [00:34<00:12, 8614.60it/s] 74%|███████▍  | 295140/400000 [00:34<00:12, 8503.64it/s] 74%|███████▍  | 295993/400000 [00:34<00:12, 8338.36it/s] 74%|███████▍  | 296863/400000 [00:34<00:12, 8443.61it/s] 74%|███████▍  | 297745/400000 [00:34<00:11, 8551.53it/s] 75%|███████▍  | 298622/400000 [00:34<00:11, 8613.40it/s] 75%|███████▍  | 299533/400000 [00:34<00:11, 8755.58it/s] 75%|███████▌  | 300410/400000 [00:34<00:11, 8664.12it/s] 75%|███████▌  | 301278/400000 [00:34<00:11, 8395.44it/s] 76%|███████▌  | 302202/400000 [00:35<00:11, 8630.37it/s] 76%|███████▌  | 303109/400000 [00:35<00:11, 8756.14it/s] 76%|███████▌  | 303988/400000 [00:35<00:11, 8696.99it/s] 76%|███████▌  | 304934/400000 [00:35<00:10, 8911.49it/s] 76%|███████▋  | 305828/400000 [00:35<00:10, 8709.79it/s] 77%|███████▋  | 306760/400000 [00:35<00:10, 8884.10it/s] 77%|███████▋  | 307667/400000 [00:35<00:10, 8936.82it/s] 77%|███████▋  | 308563/400000 [00:35<00:10, 8825.08it/s] 77%|███████▋  | 309457/400000 [00:35<00:10, 8858.33it/s] 78%|███████▊  | 310391/400000 [00:35<00:09, 8997.28it/s] 78%|███████▊  | 311293/400000 [00:36<00:10, 8832.00it/s] 78%|███████▊  | 312218/400000 [00:36<00:09, 8952.61it/s] 78%|███████▊  | 313181/400000 [00:36<00:09, 9145.29it/s] 79%|███████▊  | 314098/400000 [00:36<00:09, 9132.60it/s] 79%|███████▉  | 315023/400000 [00:36<00:09, 9166.80it/s] 79%|███████▉  | 315981/400000 [00:36<00:09, 9286.53it/s] 79%|███████▉  | 316911/400000 [00:36<00:09, 8823.72it/s] 79%|███████▉  | 317811/400000 [00:36<00:09, 8874.31it/s] 80%|███████▉  | 318703/400000 [00:36<00:09, 8766.43it/s] 80%|███████▉  | 319583/400000 [00:36<00:09, 8736.42it/s] 80%|████████  | 320536/400000 [00:37<00:08, 8958.33it/s] 80%|████████  | 321435/400000 [00:37<00:08, 8861.69it/s] 81%|████████  | 322324/400000 [00:37<00:08, 8780.98it/s] 81%|████████  | 323254/400000 [00:37<00:08, 8929.39it/s] 81%|████████  | 324163/400000 [00:37<00:08, 8976.07it/s] 81%|████████▏ | 325062/400000 [00:37<00:08, 8924.42it/s] 81%|████████▏ | 325956/400000 [00:37<00:08, 8924.61it/s] 82%|████████▏ | 326850/400000 [00:37<00:08, 8877.42it/s] 82%|████████▏ | 327744/400000 [00:37<00:08, 8893.20it/s] 82%|████████▏ | 328668/400000 [00:38<00:07, 8993.64it/s] 82%|████████▏ | 329633/400000 [00:38<00:07, 9178.58it/s] 83%|████████▎ | 330557/400000 [00:38<00:07, 9194.35it/s] 83%|████████▎ | 331522/400000 [00:38<00:07, 9323.64it/s] 83%|████████▎ | 332456/400000 [00:38<00:07, 9255.00it/s] 83%|████████▎ | 333383/400000 [00:38<00:07, 9023.23it/s] 84%|████████▎ | 334288/400000 [00:38<00:07, 8952.70it/s] 84%|████████▍ | 335185/400000 [00:38<00:07, 8675.76it/s] 84%|████████▍ | 336107/400000 [00:38<00:07, 8831.73it/s] 84%|████████▍ | 337010/400000 [00:38<00:07, 8887.78it/s] 84%|████████▍ | 337901/400000 [00:39<00:07, 8792.45it/s] 85%|████████▍ | 338815/400000 [00:39<00:06, 8891.41it/s] 85%|████████▍ | 339706/400000 [00:39<00:06, 8883.10it/s] 85%|████████▌ | 340610/400000 [00:39<00:06, 8928.54it/s] 85%|████████▌ | 341556/400000 [00:39<00:06, 9079.37it/s] 86%|████████▌ | 342527/400000 [00:39<00:06, 9257.18it/s] 86%|████████▌ | 343455/400000 [00:39<00:06, 8769.56it/s] 86%|████████▌ | 344348/400000 [00:39<00:06, 8816.95it/s] 86%|████████▋ | 345293/400000 [00:39<00:06, 8997.54it/s] 87%|████████▋ | 346197/400000 [00:39<00:05, 8989.03it/s] 87%|████████▋ | 347118/400000 [00:40<00:05, 9054.16it/s] 87%|████████▋ | 348026/400000 [00:40<00:05, 8856.25it/s] 87%|████████▋ | 348915/400000 [00:40<00:05, 8856.82it/s] 87%|████████▋ | 349803/400000 [00:40<00:05, 8839.41it/s] 88%|████████▊ | 350736/400000 [00:40<00:05, 8980.29it/s] 88%|████████▊ | 351647/400000 [00:40<00:05, 9017.91it/s] 88%|████████▊ | 352550/400000 [00:40<00:05, 9009.39it/s] 88%|████████▊ | 353456/400000 [00:40<00:05, 9024.30it/s] 89%|████████▊ | 354359/400000 [00:40<00:05, 8804.04it/s] 89%|████████▉ | 355248/400000 [00:40<00:05, 8828.35it/s] 89%|████████▉ | 356132/400000 [00:41<00:05, 8540.34it/s] 89%|████████▉ | 357029/400000 [00:41<00:04, 8660.59it/s] 89%|████████▉ | 357922/400000 [00:41<00:04, 8737.93it/s] 90%|████████▉ | 358821/400000 [00:41<00:04, 8809.48it/s] 90%|████████▉ | 359725/400000 [00:41<00:04, 8877.16it/s] 90%|█████████ | 360614/400000 [00:41<00:04, 8877.29it/s] 90%|█████████ | 361526/400000 [00:41<00:04, 8947.48it/s] 91%|█████████ | 362427/400000 [00:41<00:04, 8964.40it/s] 91%|█████████ | 363324/400000 [00:41<00:04, 8713.56it/s] 91%|█████████ | 364198/400000 [00:42<00:04, 8715.08it/s] 91%|█████████▏| 365072/400000 [00:42<00:04, 8722.14it/s] 91%|█████████▏| 365946/400000 [00:42<00:03, 8714.93it/s] 92%|█████████▏| 366819/400000 [00:42<00:03, 8677.76it/s] 92%|█████████▏| 367688/400000 [00:42<00:03, 8625.26it/s] 92%|█████████▏| 368551/400000 [00:42<00:03, 8600.71it/s] 92%|█████████▏| 369467/400000 [00:42<00:03, 8758.62it/s] 93%|█████████▎| 370344/400000 [00:42<00:03, 8607.32it/s] 93%|█████████▎| 371206/400000 [00:42<00:03, 8192.38it/s] 93%|█████████▎| 372031/400000 [00:42<00:03, 8204.17it/s] 93%|█████████▎| 372855/400000 [00:43<00:03, 8138.83it/s] 93%|█████████▎| 373672/400000 [00:43<00:03, 8109.57it/s] 94%|█████████▎| 374551/400000 [00:43<00:03, 8300.67it/s] 94%|█████████▍| 375452/400000 [00:43<00:02, 8499.36it/s] 94%|█████████▍| 376354/400000 [00:43<00:02, 8646.63it/s] 94%|█████████▍| 377241/400000 [00:43<00:02, 8709.58it/s] 95%|█████████▍| 378114/400000 [00:43<00:02, 8678.13it/s] 95%|█████████▍| 378995/400000 [00:43<00:02, 8716.47it/s] 95%|█████████▍| 379874/400000 [00:43<00:02, 8735.98it/s] 95%|█████████▌| 380778/400000 [00:43<00:02, 8824.46it/s] 95%|█████████▌| 381662/400000 [00:44<00:02, 8701.45it/s] 96%|█████████▌| 382555/400000 [00:44<00:01, 8768.56it/s] 96%|█████████▌| 383450/400000 [00:44<00:01, 8820.49it/s] 96%|█████████▌| 384346/400000 [00:44<00:01, 8861.01it/s] 96%|█████████▋| 385233/400000 [00:44<00:01, 8794.89it/s] 97%|█████████▋| 386113/400000 [00:44<00:01, 8711.75it/s] 97%|█████████▋| 387025/400000 [00:44<00:01, 8828.94it/s] 97%|█████████▋| 387934/400000 [00:44<00:01, 8905.07it/s] 97%|█████████▋| 388827/400000 [00:44<00:01, 8911.59it/s] 97%|█████████▋| 389719/400000 [00:44<00:01, 8659.97it/s] 98%|█████████▊| 390587/400000 [00:45<00:01, 8503.25it/s] 98%|█████████▊| 391444/400000 [00:45<00:01, 8522.82it/s] 98%|█████████▊| 392305/400000 [00:45<00:00, 8547.71it/s] 98%|█████████▊| 393161/400000 [00:45<00:00, 8388.73it/s] 99%|█████████▊| 394002/400000 [00:45<00:00, 8294.92it/s] 99%|█████████▊| 394841/400000 [00:45<00:00, 8320.93it/s] 99%|█████████▉| 395715/400000 [00:45<00:00, 8439.85it/s] 99%|█████████▉| 396581/400000 [00:45<00:00, 8504.61it/s] 99%|█████████▉| 397466/400000 [00:45<00:00, 8603.43it/s]100%|█████████▉| 398328/400000 [00:45<00:00, 8501.40it/s]100%|█████████▉| 399211/400000 [00:46<00:00, 8596.65it/s]100%|█████████▉| 399999/400000 [00:46<00:00, 8663.98it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7fb9e9ea6128>, <torchtext.data.dataset.TabularDataset object at 0x7fb9e9ea6278>, <torchtext.vocab.Vocab object at 0x7fb9e9ea6198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.87 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.87 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.87 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.95 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.95 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.74 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.83 file/s]2020-06-10 18:18:08.961294: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-10 18:18:08.966389: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095125000 Hz
2020-06-10 18:18:08.966552: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ce82733610 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-10 18:18:08.966567: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist_dataset_small.npy', 'test', 'fashion-mnist_small.npy', 'mnist2'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 39%|███▊      | 3833856/9912422 [00:00<00:00, 37960467.12it/s]9920512it [00:00, 36655666.49it/s]                             
0it [00:00, ?it/s]32768it [00:00, 734566.99it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160247.64it/s]1654784it [00:00, 12375838.32it/s]                         
0it [00:00, ?it/s]8192it [00:00, 237707.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

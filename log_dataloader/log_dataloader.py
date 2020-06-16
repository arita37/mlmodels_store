
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '525b5a33511bef1b32994ba341cc4ff09d4ffc92', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/525b5a33511bef1b32994ba341cc4ff09d4ffc92

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/525b5a33511bef1b32994ba341cc4ff09d4ffc92

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f688be956a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f688be956a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f68f745d0d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f68f745d0d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f6911189ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f6911189ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f68a4cd8950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f68a4cd8950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f68a4cd8950> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:12, 136976.79it/s] 79%|███████▉  | 7847936/9912422 [00:00<00:10, 195534.11it/s]9920512it [00:00, 42086099.90it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1282713.97it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1018379.67it/s]1654784it [00:00, 12383301.55it/s]                           
0it [00:00, ?it/s]8192it [00:00, 254012.32it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f688bd72ba8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f688bd64e48>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f68a4cd8598> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f68a4cd8598> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f68a4cd8598> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:33,  1.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:33,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:33,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:32,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:31,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:31,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:30,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:30,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:29,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:28,  1.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 10/162 [00:00<01:02,  2.43 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:02,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:02,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:01,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:01,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:00,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:00,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:59,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:59,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:59,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:58,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:58,  2.43 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:40,  3.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:40,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:40,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:40,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:40,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:39,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:39,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:39,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:38,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:38,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:38,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:38,  3.44 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|█▉        | 32/162 [00:00<00:26,  4.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:26,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:26,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:26,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:26,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:25,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:25,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:25,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:25,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:25,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:24,  4.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:00<00:17,  6.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:17,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:16,  6.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:11,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:10,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:10,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:10,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:10,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:10,  9.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.99 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:07, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:06, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:06, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:06, 12.99 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:04, 17.63 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:03, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:03, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:03, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 23.45 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.46 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 30.46 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 38.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:01, 38.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 46.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 54.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 54.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 61.87 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 61.87 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.10 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.10 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.16 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.27s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 68.16 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.76 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.53s/ url]
0 examples [00:00, ? examples/s]2020-06-16 06:09:48.003786: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-16 06:09:48.067202: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095139999 Hz
2020-06-16 06:09:48.067398: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x555777f5f420 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-16 06:09:48.067417: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.89 examples/s]104 examples [00:00,  2.70 examples/s]210 examples [00:00,  3.85 examples/s]315 examples [00:00,  5.50 examples/s]420 examples [00:00,  7.84 examples/s]527 examples [00:01, 11.16 examples/s]632 examples [00:01, 15.87 examples/s]736 examples [00:01, 22.52 examples/s]840 examples [00:01, 31.88 examples/s]941 examples [00:01, 44.94 examples/s]1047 examples [00:01, 63.04 examples/s]1151 examples [00:01, 87.76 examples/s]1257 examples [00:01, 121.08 examples/s]1363 examples [00:01, 164.87 examples/s]1467 examples [00:01, 220.38 examples/s]1574 examples [00:02, 289.18 examples/s]1679 examples [00:02, 369.35 examples/s]1784 examples [00:02, 457.75 examples/s]1889 examples [00:02, 550.17 examples/s]1994 examples [00:02, 641.70 examples/s]2099 examples [00:02, 726.10 examples/s]2205 examples [00:02, 800.24 examples/s]2310 examples [00:02, 851.29 examples/s]2414 examples [00:02, 887.06 examples/s]2516 examples [00:02, 906.40 examples/s]2622 examples [00:03, 946.15 examples/s]2727 examples [00:03, 974.93 examples/s]2833 examples [00:03, 998.58 examples/s]2939 examples [00:03, 1013.62 examples/s]3044 examples [00:03, 982.19 examples/s] 3148 examples [00:03, 996.57 examples/s]3254 examples [00:03, 1012.39 examples/s]3360 examples [00:03, 1024.19 examples/s]3466 examples [00:03, 1032.62 examples/s]3572 examples [00:03, 1039.42 examples/s]3678 examples [00:04, 1044.26 examples/s]3784 examples [00:04, 1046.36 examples/s]3891 examples [00:04, 1051.51 examples/s]3998 examples [00:04, 1055.88 examples/s]4104 examples [00:04, 1056.27 examples/s]4210 examples [00:04, 1050.10 examples/s]4316 examples [00:04, 1046.61 examples/s]4424 examples [00:04, 1053.95 examples/s]4530 examples [00:04, 1052.84 examples/s]4636 examples [00:04, 1044.33 examples/s]4743 examples [00:05, 1050.39 examples/s]4849 examples [00:05, 1044.39 examples/s]4954 examples [00:05, 1022.26 examples/s]5060 examples [00:05, 1032.58 examples/s]5166 examples [00:05, 1039.97 examples/s]5272 examples [00:05, 1045.16 examples/s]5377 examples [00:05, 1031.92 examples/s]5481 examples [00:05, 1032.39 examples/s]5585 examples [00:05, 1031.55 examples/s]5689 examples [00:06, 1020.95 examples/s]5792 examples [00:06, 1011.68 examples/s]5894 examples [00:06, 999.72 examples/s] 5996 examples [00:06, 1004.85 examples/s]6099 examples [00:06, 1009.15 examples/s]6200 examples [00:06, 1005.15 examples/s]6304 examples [00:06, 1013.49 examples/s]6410 examples [00:06, 1026.22 examples/s]6513 examples [00:06, 1008.07 examples/s]6614 examples [00:06, 992.93 examples/s] 6718 examples [00:07, 1004.65 examples/s]6821 examples [00:07, 1011.94 examples/s]6926 examples [00:07, 1022.32 examples/s]7029 examples [00:07, 1015.51 examples/s]7133 examples [00:07, 1020.78 examples/s]7239 examples [00:07, 1029.88 examples/s]7343 examples [00:07, 1030.75 examples/s]7447 examples [00:07, 1028.60 examples/s]7550 examples [00:07, 1021.97 examples/s]7653 examples [00:07, 1009.22 examples/s]7757 examples [00:08, 1016.32 examples/s]7862 examples [00:08, 1024.33 examples/s]7968 examples [00:08, 1032.24 examples/s]8072 examples [00:08, 1026.83 examples/s]8175 examples [00:08, 1017.49 examples/s]8277 examples [00:08, 999.92 examples/s] 8380 examples [00:08, 1006.99 examples/s]8485 examples [00:08, 1018.18 examples/s]8589 examples [00:08, 1022.19 examples/s]8693 examples [00:08, 1025.24 examples/s]8797 examples [00:09, 1027.85 examples/s]8902 examples [00:09, 1031.69 examples/s]9007 examples [00:09, 1036.04 examples/s]9111 examples [00:09, 1031.59 examples/s]9215 examples [00:09, 1030.41 examples/s]9319 examples [00:09, 1032.66 examples/s]9424 examples [00:09, 1036.92 examples/s]9530 examples [00:09, 1041.94 examples/s]9635 examples [00:09, 1032.84 examples/s]9740 examples [00:09, 1035.61 examples/s]9845 examples [00:10, 1037.19 examples/s]9949 examples [00:10, 1014.03 examples/s]10051 examples [00:10, 970.94 examples/s]10153 examples [00:10, 985.12 examples/s]10252 examples [00:10, 957.74 examples/s]10357 examples [00:10, 983.46 examples/s]10456 examples [00:10, 972.98 examples/s]10554 examples [00:10, 964.59 examples/s]10655 examples [00:10, 975.74 examples/s]10753 examples [00:11, 975.30 examples/s]10857 examples [00:11, 991.28 examples/s]10963 examples [00:11, 1009.14 examples/s]11067 examples [00:11, 1016.81 examples/s]11170 examples [00:11, 1020.40 examples/s]11273 examples [00:11, 1003.80 examples/s]11374 examples [00:11, 1003.58 examples/s]11480 examples [00:11, 1018.41 examples/s]11585 examples [00:11, 1027.67 examples/s]11688 examples [00:11, 1026.45 examples/s]11791 examples [00:12, 1011.90 examples/s]11897 examples [00:12, 1024.98 examples/s]12002 examples [00:12, 1030.68 examples/s]12107 examples [00:12, 1034.12 examples/s]12211 examples [00:12, 1027.15 examples/s]12314 examples [00:12, 1024.60 examples/s]12419 examples [00:12, 1030.56 examples/s]12525 examples [00:12, 1036.57 examples/s]12632 examples [00:12, 1045.03 examples/s]12737 examples [00:12, 1041.98 examples/s]12842 examples [00:13, 1040.23 examples/s]12948 examples [00:13, 1045.27 examples/s]13054 examples [00:13, 1047.28 examples/s]13160 examples [00:13, 1049.57 examples/s]13266 examples [00:13, 1050.86 examples/s]13372 examples [00:13, 1041.97 examples/s]13477 examples [00:13, 1038.30 examples/s]13581 examples [00:13, 1036.32 examples/s]13686 examples [00:13, 1037.80 examples/s]13792 examples [00:13, 1042.40 examples/s]13897 examples [00:14, 1014.98 examples/s]14001 examples [00:14, 1020.70 examples/s]14107 examples [00:14, 1029.75 examples/s]14211 examples [00:14, 1029.68 examples/s]14315 examples [00:14, 1000.27 examples/s]14416 examples [00:14, 999.40 examples/s] 14517 examples [00:14, 991.58 examples/s]14618 examples [00:14, 997.03 examples/s]14723 examples [00:14, 1011.62 examples/s]14826 examples [00:14, 1014.34 examples/s]14930 examples [00:15, 1020.02 examples/s]15037 examples [00:15, 1032.23 examples/s]15142 examples [00:15, 1036.11 examples/s]15247 examples [00:15, 1037.66 examples/s]15351 examples [00:15, 1022.98 examples/s]15454 examples [00:15, 1007.39 examples/s]15556 examples [00:15, 1009.40 examples/s]15662 examples [00:15, 1023.78 examples/s]15768 examples [00:15, 1032.77 examples/s]15874 examples [00:16, 1039.97 examples/s]15979 examples [00:16, 1028.59 examples/s]16085 examples [00:16, 1036.76 examples/s]16191 examples [00:16, 1043.54 examples/s]16298 examples [00:16, 1049.70 examples/s]16404 examples [00:16, 1045.96 examples/s]16509 examples [00:16, 1043.52 examples/s]16615 examples [00:16, 1045.77 examples/s]16721 examples [00:16, 1049.28 examples/s]16826 examples [00:16, 1048.71 examples/s]16931 examples [00:17, 1047.74 examples/s]17036 examples [00:17, 1019.43 examples/s]17139 examples [00:17, 1020.15 examples/s]17245 examples [00:17, 1029.82 examples/s]17351 examples [00:17, 1036.12 examples/s]17458 examples [00:17, 1043.79 examples/s]17563 examples [00:17, 1042.88 examples/s]17668 examples [00:17, 1043.12 examples/s]17775 examples [00:17, 1048.16 examples/s]17880 examples [00:17, 1045.80 examples/s]17987 examples [00:18, 1051.94 examples/s]18093 examples [00:18, 1049.39 examples/s]18198 examples [00:18, 1046.14 examples/s]18303 examples [00:18, 1045.47 examples/s]18408 examples [00:18, 1042.60 examples/s]18514 examples [00:18, 1047.60 examples/s]18619 examples [00:18, 1042.75 examples/s]18724 examples [00:18, 1025.50 examples/s]18827 examples [00:18, 1010.40 examples/s]18933 examples [00:18, 1022.54 examples/s]19039 examples [00:19, 1031.85 examples/s]19143 examples [00:19, 1005.69 examples/s]19250 examples [00:19, 1022.16 examples/s]19355 examples [00:19, 1029.85 examples/s]19461 examples [00:19, 1038.28 examples/s]19567 examples [00:19, 1043.54 examples/s]19672 examples [00:19, 1045.19 examples/s]19778 examples [00:19, 1046.96 examples/s]19884 examples [00:19, 1048.32 examples/s]19989 examples [00:19, 1040.55 examples/s]20094 examples [00:20, 988.92 examples/s] 20194 examples [00:20, 989.34 examples/s]20294 examples [00:20, 988.68 examples/s]20400 examples [00:20, 1006.88 examples/s]20506 examples [00:20, 1020.92 examples/s]20612 examples [00:20, 1029.39 examples/s]20716 examples [00:20, 1029.23 examples/s]20820 examples [00:20, 1003.11 examples/s]20926 examples [00:20, 1017.86 examples/s]21029 examples [00:20, 1018.61 examples/s]21134 examples [00:21, 1027.52 examples/s]21237 examples [00:21, 1026.55 examples/s]21341 examples [00:21, 1030.08 examples/s]21445 examples [00:21, 1023.55 examples/s]21548 examples [00:21, 1024.03 examples/s]21655 examples [00:21, 1037.11 examples/s]21760 examples [00:21, 1039.36 examples/s]21864 examples [00:21, 1038.36 examples/s]21971 examples [00:21, 1047.55 examples/s]22077 examples [00:22, 1049.79 examples/s]22183 examples [00:22, 1051.32 examples/s]22289 examples [00:22, 1040.25 examples/s]22395 examples [00:22, 1044.65 examples/s]22501 examples [00:22, 1048.38 examples/s]22608 examples [00:22, 1053.75 examples/s]22715 examples [00:22, 1056.61 examples/s]22821 examples [00:22, 1050.87 examples/s]22927 examples [00:22, 1046.67 examples/s]23033 examples [00:22, 1048.20 examples/s]23138 examples [00:23, 1048.71 examples/s]23244 examples [00:23, 1050.57 examples/s]23350 examples [00:23, 1027.99 examples/s]23453 examples [00:23, 1006.62 examples/s]23559 examples [00:23, 1020.49 examples/s]23665 examples [00:23, 1029.44 examples/s]23769 examples [00:23, 1016.47 examples/s]23873 examples [00:23, 1022.48 examples/s]23978 examples [00:23, 1029.05 examples/s]24083 examples [00:23, 1034.99 examples/s]24187 examples [00:24, 1032.37 examples/s]24293 examples [00:24, 1039.17 examples/s]24398 examples [00:24, 1039.89 examples/s]24503 examples [00:24, 1035.28 examples/s]24608 examples [00:24, 1039.44 examples/s]24713 examples [00:24, 1040.41 examples/s]24819 examples [00:24, 1043.76 examples/s]24924 examples [00:24, 1043.68 examples/s]25029 examples [00:24, 1019.27 examples/s]25133 examples [00:24, 1022.71 examples/s]25236 examples [00:25, 1014.83 examples/s]25340 examples [00:25, 1022.25 examples/s]25445 examples [00:25, 1028.72 examples/s]25549 examples [00:25, 1031.31 examples/s]25654 examples [00:25, 1034.56 examples/s]25759 examples [00:25, 1038.93 examples/s]25863 examples [00:25, 1039.04 examples/s]25967 examples [00:25, 1037.92 examples/s]26071 examples [00:25, 1013.82 examples/s]26176 examples [00:25, 1023.26 examples/s]26280 examples [00:26, 1026.83 examples/s]26383 examples [00:26, 1026.77 examples/s]26486 examples [00:26, 1025.04 examples/s]26589 examples [00:26, 1010.03 examples/s]26691 examples [00:26, 1008.17 examples/s]26798 examples [00:26, 1024.17 examples/s]26902 examples [00:26, 1028.69 examples/s]27005 examples [00:26, 1023.77 examples/s]27110 examples [00:26, 1028.99 examples/s]27213 examples [00:26, 1028.56 examples/s]27316 examples [00:27, 1028.88 examples/s]27421 examples [00:27, 1032.70 examples/s]27525 examples [00:27, 1031.06 examples/s]27629 examples [00:27, 1029.30 examples/s]27732 examples [00:27, 1025.16 examples/s]27836 examples [00:27, 1028.23 examples/s]27941 examples [00:27, 1033.99 examples/s]28045 examples [00:27, 1030.92 examples/s]28153 examples [00:27, 1042.65 examples/s]28258 examples [00:27, 1036.02 examples/s]28362 examples [00:28, 1009.14 examples/s]28468 examples [00:28, 1022.54 examples/s]28571 examples [00:28, 988.04 examples/s] 28673 examples [00:28, 995.48 examples/s]28775 examples [00:28, 1001.39 examples/s]28880 examples [00:28, 1014.27 examples/s]28984 examples [00:28, 1019.65 examples/s]29088 examples [00:28, 1022.95 examples/s]29193 examples [00:28, 1030.46 examples/s]29298 examples [00:29, 1035.95 examples/s]29402 examples [00:29, 1033.23 examples/s]29508 examples [00:29, 1040.42 examples/s]29613 examples [00:29, 1036.90 examples/s]29717 examples [00:29, 999.46 examples/s] 29822 examples [00:29, 1011.58 examples/s]29927 examples [00:29, 1021.63 examples/s]30030 examples [00:29, 978.98 examples/s] 30134 examples [00:29, 994.82 examples/s]30238 examples [00:29, 1006.04 examples/s]30342 examples [00:30, 1014.66 examples/s]30445 examples [00:30, 1018.95 examples/s]30551 examples [00:30, 1029.49 examples/s]30655 examples [00:30, 991.02 examples/s] 30755 examples [00:30, 978.65 examples/s]30860 examples [00:30, 998.06 examples/s]30966 examples [00:30, 1015.84 examples/s]31070 examples [00:30, 1020.91 examples/s]31173 examples [00:30, 1015.90 examples/s]31279 examples [00:30, 1026.98 examples/s]31385 examples [00:31, 1035.52 examples/s]31490 examples [00:31, 1038.05 examples/s]31596 examples [00:31, 1044.31 examples/s]31701 examples [00:31, 1043.85 examples/s]31806 examples [00:31, 1032.10 examples/s]31912 examples [00:31, 1039.48 examples/s]32018 examples [00:31, 1043.19 examples/s]32124 examples [00:31, 1047.84 examples/s]32230 examples [00:31, 1048.42 examples/s]32336 examples [00:31, 1049.43 examples/s]32443 examples [00:32, 1052.72 examples/s]32549 examples [00:32, 1052.25 examples/s]32656 examples [00:32, 1055.82 examples/s]32762 examples [00:32, 1053.06 examples/s]32868 examples [00:32, 974.80 examples/s] 32969 examples [00:32, 984.90 examples/s]33072 examples [00:32, 996.84 examples/s]33177 examples [00:32, 1009.02 examples/s]33279 examples [00:32, 1010.70 examples/s]33383 examples [00:33, 1018.74 examples/s]33486 examples [00:33, 1019.69 examples/s]33592 examples [00:33, 1029.03 examples/s]33698 examples [00:33, 1037.65 examples/s]33803 examples [00:33, 1038.18 examples/s]33907 examples [00:33, 1034.26 examples/s]34013 examples [00:33, 1040.50 examples/s]34119 examples [00:33, 1044.93 examples/s]34224 examples [00:33, 1042.19 examples/s]34329 examples [00:33, 1043.99 examples/s]34435 examples [00:34, 1047.43 examples/s]34541 examples [00:34, 1049.94 examples/s]34647 examples [00:34, 1050.64 examples/s]34753 examples [00:34, 1050.93 examples/s]34859 examples [00:34, 1050.63 examples/s]34965 examples [00:34, 1046.94 examples/s]35070 examples [00:34, 1047.66 examples/s]35175 examples [00:34, 1046.00 examples/s]35280 examples [00:34, 1013.49 examples/s]35382 examples [00:34, 1007.45 examples/s]35487 examples [00:35, 1017.61 examples/s]35592 examples [00:35, 1027.06 examples/s]35695 examples [00:35, 1023.54 examples/s]35799 examples [00:35, 1024.46 examples/s]35903 examples [00:35, 1027.50 examples/s]36006 examples [00:35, 997.15 examples/s] 36113 examples [00:35, 1015.50 examples/s]36219 examples [00:35, 1027.17 examples/s]36325 examples [00:35, 1035.37 examples/s]36429 examples [00:35, 1021.31 examples/s]36536 examples [00:36, 1033.91 examples/s]36643 examples [00:36, 1042.32 examples/s]36748 examples [00:36, 1039.17 examples/s]36853 examples [00:36, 1037.24 examples/s]36957 examples [00:36, 1030.81 examples/s]37061 examples [00:36, 1027.09 examples/s]37164 examples [00:36, 1019.67 examples/s]37270 examples [00:36, 1029.04 examples/s]37376 examples [00:36, 1037.05 examples/s]37480 examples [00:36, 1003.10 examples/s]37581 examples [00:37, 1003.36 examples/s]37687 examples [00:37, 1018.88 examples/s]37790 examples [00:37, 1020.35 examples/s]37895 examples [00:37, 1026.79 examples/s]37999 examples [00:37, 1029.25 examples/s]38102 examples [00:37, 1022.82 examples/s]38208 examples [00:37, 1033.41 examples/s]38312 examples [00:37, 1016.98 examples/s]38414 examples [00:37, 1009.25 examples/s]38519 examples [00:38, 1020.91 examples/s]38625 examples [00:38, 1030.52 examples/s]38729 examples [00:38, 1022.95 examples/s]38834 examples [00:38, 1030.42 examples/s]38939 examples [00:38, 1035.17 examples/s]39044 examples [00:38, 1037.31 examples/s]39148 examples [00:38, 1012.03 examples/s]39254 examples [00:38, 1024.17 examples/s]39358 examples [00:38, 1028.72 examples/s]39462 examples [00:38, 1031.71 examples/s]39566 examples [00:39, 1027.11 examples/s]39670 examples [00:39, 1030.92 examples/s]39774 examples [00:39, 1021.17 examples/s]39877 examples [00:39, 989.49 examples/s] 39983 examples [00:39, 1007.18 examples/s]40084 examples [00:39, 962.40 examples/s] 40190 examples [00:39, 987.29 examples/s]40296 examples [00:39, 1005.85 examples/s]40402 examples [00:39, 1020.41 examples/s]40506 examples [00:39, 1024.40 examples/s]40611 examples [00:40, 1031.20 examples/s]40717 examples [00:40, 1038.67 examples/s]40823 examples [00:40, 1043.88 examples/s]40929 examples [00:40, 1044.48 examples/s]41034 examples [00:40, 1042.49 examples/s]41140 examples [00:40, 1047.06 examples/s]41246 examples [00:40, 1049.84 examples/s]41352 examples [00:40, 1047.37 examples/s]41457 examples [00:40, 1047.02 examples/s]41562 examples [00:40, 1043.06 examples/s]41668 examples [00:41, 1045.69 examples/s]41775 examples [00:41, 1049.85 examples/s]41882 examples [00:41, 1054.96 examples/s]41988 examples [00:41, 1046.47 examples/s]42093 examples [00:41, 1041.58 examples/s]42198 examples [00:41, 997.11 examples/s] 42299 examples [00:41, 989.45 examples/s]42404 examples [00:41, 1005.94 examples/s]42509 examples [00:41, 1017.49 examples/s]42611 examples [00:41, 1018.00 examples/s]42714 examples [00:42, 1019.60 examples/s]42820 examples [00:42, 1028.92 examples/s]42927 examples [00:42, 1038.32 examples/s]43032 examples [00:42, 1039.24 examples/s]43136 examples [00:42, 1024.53 examples/s]43240 examples [00:42, 1026.77 examples/s]43345 examples [00:42, 1032.69 examples/s]43449 examples [00:42, 1033.99 examples/s]43553 examples [00:42, 1028.73 examples/s]43656 examples [00:43, 1026.57 examples/s]43761 examples [00:43, 1030.86 examples/s]43866 examples [00:43, 1034.59 examples/s]43971 examples [00:43, 1038.82 examples/s]44075 examples [00:43, 1033.69 examples/s]44179 examples [00:43, 1029.98 examples/s]44283 examples [00:43, 1023.73 examples/s]44386 examples [00:43, 1016.08 examples/s]44488 examples [00:43, 1014.50 examples/s]44594 examples [00:43, 1025.40 examples/s]44697 examples [00:44, 1024.90 examples/s]44802 examples [00:44, 1030.92 examples/s]44906 examples [00:44, 1033.04 examples/s]45010 examples [00:44, 1030.57 examples/s]45116 examples [00:44, 1036.51 examples/s]45220 examples [00:44, 1027.84 examples/s]45326 examples [00:44, 1035.78 examples/s]45430 examples [00:44, 1007.78 examples/s]45535 examples [00:44, 1018.13 examples/s]45640 examples [00:44, 1026.22 examples/s]45745 examples [00:45, 1032.21 examples/s]45852 examples [00:45, 1040.89 examples/s]45958 examples [00:45, 1046.01 examples/s]46063 examples [00:45, 1030.56 examples/s]46168 examples [00:45, 1035.40 examples/s]46272 examples [00:45, 1028.13 examples/s]46375 examples [00:45, 1023.52 examples/s]46478 examples [00:45, 1019.84 examples/s]46583 examples [00:45, 1027.26 examples/s]46688 examples [00:45, 1031.33 examples/s]46792 examples [00:46, 1002.15 examples/s]46894 examples [00:46, 1006.29 examples/s]46999 examples [00:46, 1016.96 examples/s]47104 examples [00:46, 1024.42 examples/s]47207 examples [00:46, 1025.72 examples/s]47310 examples [00:46, 1018.12 examples/s]47412 examples [00:46, 1017.44 examples/s]47519 examples [00:46, 1029.96 examples/s]47625 examples [00:46, 1036.55 examples/s]47729 examples [00:46, 1034.75 examples/s]47833 examples [00:47, 1036.19 examples/s]47939 examples [00:47, 1040.54 examples/s]48045 examples [00:47, 1046.13 examples/s]48150 examples [00:47, 1046.31 examples/s]48255 examples [00:47, 1045.12 examples/s]48360 examples [00:47, 1042.67 examples/s]48465 examples [00:47, 1040.69 examples/s]48570 examples [00:47, 1022.32 examples/s]48673 examples [00:47, 1022.93 examples/s]48778 examples [00:47, 1028.30 examples/s]48882 examples [00:48, 1031.39 examples/s]48987 examples [00:48, 1036.46 examples/s]49091 examples [00:48, 1035.76 examples/s]49195 examples [00:48, 1007.08 examples/s]49300 examples [00:48, 1017.11 examples/s]49404 examples [00:48, 1022.52 examples/s]49508 examples [00:48, 1026.62 examples/s]49613 examples [00:48, 1032.92 examples/s]49717 examples [00:48, 1032.16 examples/s]49821 examples [00:48, 1033.60 examples/s]49925 examples [00:49, 1031.11 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6258/50000 [00:00<00:00, 62568.13 examples/s] 37%|███▋      | 18370/50000 [00:00<00:00, 73180.90 examples/s] 62%|██████▏   | 31069/50000 [00:00<00:00, 83838.12 examples/s] 87%|████████▋ | 43533/50000 [00:00<00:00, 92966.37 examples/s]                                                               0 examples [00:00, ? examples/s]77 examples [00:00, 763.11 examples/s]162 examples [00:00, 787.09 examples/s]263 examples [00:00, 841.73 examples/s]370 examples [00:00, 897.69 examples/s]476 examples [00:00, 939.81 examples/s]582 examples [00:00, 971.27 examples/s]687 examples [00:00, 992.38 examples/s]792 examples [00:00, 1007.28 examples/s]895 examples [00:00, 1012.68 examples/s]999 examples [00:01, 1018.29 examples/s]1102 examples [00:01, 1020.54 examples/s]1204 examples [00:01, 1020.33 examples/s]1306 examples [00:01, 1008.73 examples/s]1407 examples [00:01, 797.92 examples/s] 1514 examples [00:01, 863.67 examples/s]1620 examples [00:01, 914.36 examples/s]1717 examples [00:01, 925.60 examples/s]1815 examples [00:01, 941.03 examples/s]1916 examples [00:02, 959.16 examples/s]2020 examples [00:02, 980.25 examples/s]2120 examples [00:02, 973.14 examples/s]2225 examples [00:02, 994.71 examples/s]2330 examples [00:02, 1009.98 examples/s]2434 examples [00:02, 1018.32 examples/s]2540 examples [00:02, 1028.01 examples/s]2644 examples [00:02, 1006.56 examples/s]2750 examples [00:02, 1019.16 examples/s]2854 examples [00:02, 1022.65 examples/s]2958 examples [00:03, 1026.71 examples/s]3065 examples [00:03, 1036.74 examples/s]3169 examples [00:03, 1015.32 examples/s]3275 examples [00:03, 1026.79 examples/s]3381 examples [00:03, 1034.73 examples/s]3486 examples [00:03, 1038.37 examples/s]3594 examples [00:03, 1049.01 examples/s]3701 examples [00:03, 1052.65 examples/s]3808 examples [00:03, 1056.68 examples/s]3914 examples [00:03, 1053.98 examples/s]4021 examples [00:04, 1055.85 examples/s]4127 examples [00:04, 1056.36 examples/s]4233 examples [00:04, 1056.50 examples/s]4339 examples [00:04, 1023.09 examples/s]4445 examples [00:04, 1031.88 examples/s]4550 examples [00:04, 1037.17 examples/s]4654 examples [00:04, 1025.83 examples/s]4759 examples [00:04, 1031.71 examples/s]4865 examples [00:04, 1039.74 examples/s]4972 examples [00:04, 1045.91 examples/s]5077 examples [00:05, 1025.43 examples/s]5184 examples [00:05, 1036.92 examples/s]5288 examples [00:05, 1035.83 examples/s]5392 examples [00:05, 1027.36 examples/s]5496 examples [00:05, 1030.25 examples/s]5601 examples [00:05, 1033.34 examples/s]5707 examples [00:05, 1041.16 examples/s]5812 examples [00:05, 1036.95 examples/s]5916 examples [00:05, 1030.98 examples/s]6020 examples [00:05, 1033.46 examples/s]6124 examples [00:06, 1012.38 examples/s]6230 examples [00:06, 1023.88 examples/s]6334 examples [00:06, 1026.69 examples/s]6439 examples [00:06, 1031.85 examples/s]6545 examples [00:06, 1037.28 examples/s]6651 examples [00:06, 1043.17 examples/s]6758 examples [00:06, 1050.15 examples/s]6864 examples [00:06, 1051.83 examples/s]6970 examples [00:06, 1054.24 examples/s]7076 examples [00:06, 1031.29 examples/s]7181 examples [00:07, 1036.57 examples/s]7287 examples [00:07, 1042.73 examples/s]7393 examples [00:07, 1046.91 examples/s]7498 examples [00:07, 1003.92 examples/s]7601 examples [00:07, 1011.24 examples/s]7703 examples [00:07, 1011.27 examples/s]7809 examples [00:07, 1024.83 examples/s]7913 examples [00:07, 1027.51 examples/s]8016 examples [00:07, 1020.82 examples/s]8119 examples [00:08, 1003.11 examples/s]8225 examples [00:08, 1017.78 examples/s]8332 examples [00:08, 1030.22 examples/s]8438 examples [00:08, 1037.01 examples/s]8544 examples [00:08, 1043.51 examples/s]8650 examples [00:08, 1047.64 examples/s]8755 examples [00:08, 985.95 examples/s] 8861 examples [00:08, 1005.43 examples/s]8963 examples [00:08, 1001.37 examples/s]9066 examples [00:08, 1009.49 examples/s]9172 examples [00:09, 1021.92 examples/s]9279 examples [00:09, 1034.45 examples/s]9386 examples [00:09, 1043.14 examples/s]9494 examples [00:09, 1051.56 examples/s]9601 examples [00:09, 1055.11 examples/s]9707 examples [00:09, 1053.97 examples/s]9813 examples [00:09, 1053.69 examples/s]9919 examples [00:09, 1054.75 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTND8BJ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteTND8BJ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f68a4cd8950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f68a4cd8950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f68a4cd8950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f690af65b38>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f682a190160>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f68a4cd8950> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f68a4cd8950> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f68a4cd8950> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f688b098128>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f688b098080>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f681c6a2378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f681c6a2378> 

  function with postional parmater data_info <function split_train_valid at 0x7f681c6a2378> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f681c6a2488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f681c6a2488> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f681c6a2488> , (data_info, **args) 

  Download en 
Collecting en_core_web_sm==2.2.5
  Downloading https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz (12.0 MB)
Requirement already satisfied: spacy>=2.2.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from en_core_web_sm==2.2.5) (2.2.4)
Requirement already satisfied: setuptools in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (45.2.0)
Requirement already satisfied: murmurhash<1.1.0,>=0.28.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: srsly<1.1.0,>=1.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.2)
Requirement already satisfied: blis<0.5.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.4.1)
Requirement already satisfied: requests<3.0.0,>=2.13.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.23.0)
Requirement already satisfied: numpy>=1.15.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.18.5)
Requirement already satisfied: catalogue<1.1.0,>=0.0.7 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.0.0)
Requirement already satisfied: preshed<3.1.0,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.2)
Requirement already satisfied: cymem<2.1.0,>=2.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (2.0.3)
Requirement already satisfied: wasabi<1.1.0,>=0.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (0.6.0)
Requirement already satisfied: tqdm<5.0.0,>=4.38.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (4.46.1)
Requirement already satisfied: thinc==7.4.0 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (7.4.0)
Requirement already satisfied: plac<1.2.0,>=0.9.6 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from spacy>=2.2.2->en_core_web_sm==2.2.5) (1.1.3)
Requirement already satisfied: certifi>=2017.4.17 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2020.4.5.2)
Requirement already satisfied: idna<3,>=2.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (2.9)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.25.9)
Requirement already satisfied: chardet<4,>=3.0.2 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from requests<3.0.0,>=2.13.0->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.0.4)
Requirement already satisfied: importlib-metadata>=0.20; python_version < "3.8" in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (1.6.1)
Requirement already satisfied: zipp>=0.5 in /opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages (from importlib-metadata>=0.20; python_version < "3.8"->catalogue<1.1.0,>=0.0.7->spacy>=2.2.2->en_core_web_sm==2.2.5) (3.1.0)
Building wheels for collected packages: en-core-web-sm
  Building wheel for en-core-web-sm (setup.py): started
  Building wheel for en-core-web-sm (setup.py): finished with status 'done'
  Created wheel for en-core-web-sm: filename=en_core_web_sm-2.2.5-py3-none-any.whl size=12011738 sha256=54b3e5f06ad6842c0899d294747cbf0ea17a74ba6cee316f366e304efde902d3
  Stored in directory: /tmp/pip-ephem-wheel-cache-nvhqpqux/wheels/b5/94/56/596daa677d7e91038cbddfcf32b591d0c915a1b3a3e3d3c79d
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
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<24:45:05, 9.68kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<17:33:49, 13.6kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:01<12:21:00, 19.4kB/s] .vector_cache/glove.6B.zip:   0%|          | 877k/862M [00:01<8:38:58, 27.7kB/s] .vector_cache/glove.6B.zip:   0%|          | 1.79M/862M [00:01<6:03:23, 39.5kB/s].vector_cache/glove.6B.zip:   0%|          | 2.34M/862M [00:01<4:16:40, 55.8kB/s].vector_cache/glove.6B.zip:   1%|          | 5.17M/862M [00:01<2:59:13, 79.7kB/s].vector_cache/glove.6B.zip:   1%|          | 8.90M/862M [00:01<2:05:02, 114kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 12.9M/862M [00:01<1:27:12, 162kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.5M/862M [00:02<1:00:49, 231kB/s].vector_cache/glove.6B.zip:   2%|▏         | 19.2M/862M [00:02<42:45, 329kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 24.0M/862M [00:02<29:51, 468kB/s].vector_cache/glove.6B.zip:   3%|▎         | 29.2M/862M [00:02<20:51, 666kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:02<14:41, 940kB/s].vector_cache/glove.6B.zip:   4%|▍         | 37.0M/862M [00:02<10:20, 1.33MB/s].vector_cache/glove.6B.zip:   5%|▍         | 39.6M/862M [00:02<07:22, 1.86MB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.2M/862M [00:02<05:18, 2.57MB/s].vector_cache/glove.6B.zip:   5%|▌         | 44.5M/862M [00:02<03:52, 3.51MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.2M/862M [00:03<02:51, 4.74MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.1M/862M [00:03<02:06, 6.44MB/s].vector_cache/glove.6B.zip:   6%|▌         | 52.4M/862M [00:03<01:46, 7.58MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.4M/862M [00:05<03:44, 3.60MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.5M/862M [00:05<06:23, 2.10MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.1M/862M [00:05<05:23, 2.49MB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.5M/862M [00:05<04:03, 3.30MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:07<05:56, 2.25MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.9M/862M [00:07<06:04, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.9M/862M [00:07<04:38, 2.88MB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.5M/862M [00:07<03:30, 3.80MB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:09<07:57, 1.67MB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.0M/862M [00:09<07:28, 1.78MB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.2M/862M [00:09<05:38, 2.36MB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.9M/862M [00:11<06:27, 2.05MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:11<06:05, 2.17MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.7M/862M [00:11<05:00, 2.64MB/s].vector_cache/glove.6B.zip:   8%|▊         | 70.4M/862M [00:11<03:43, 3.54MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.0M/862M [00:13<06:47, 1.94MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.3M/862M [00:13<06:37, 1.99MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.4M/862M [00:13<05:06, 2.57MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.1M/862M [00:15<06:02, 2.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:15<07:12, 1.82MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.0M/862M [00:15<05:46, 2.27MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.9M/862M [00:15<04:10, 3.12MB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.3M/862M [00:17<17:57, 725kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.5M/862M [00:17<15:32, 838kB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.2M/862M [00:17<11:36, 1.12MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.2M/862M [00:17<08:18, 1.56MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:18<10:51, 1.19MB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:19<10:39, 1.22MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.4M/862M [00:19<08:11, 1.58MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.0M/862M [00:19<05:52, 2.20MB/s].vector_cache/glove.6B.zip:  10%|█         | 88.7M/862M [00:20<13:42, 940kB/s] .vector_cache/glove.6B.zip:  10%|█         | 88.8M/862M [00:21<12:35, 1.02MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.6M/862M [00:21<09:26, 1.36MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.7M/862M [00:21<06:55, 1.86MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.8M/862M [00:22<07:47, 1.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:23<08:16, 1.55MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:23<06:23, 2.00MB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:23<04:51, 2.64MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:24<06:13, 2.05MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.1M/862M [00:25<07:16, 1.75MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.8M/862M [00:25<05:49, 2.19MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 100M/862M [00:25<04:14, 2.99MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:30, 1.49MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:27<08:34, 1.48MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:27<06:34, 1.93MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:27<04:46, 2.65MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<08:49, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:29<08:50, 1.43MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:29<06:45, 1.86MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:29<05:03, 2.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:30<06:30, 1.93MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<07:32, 1.66MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:31<05:51, 2.14MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:31<04:29, 2.79MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:46, 2.16MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<06:40, 1.87MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:33<05:45, 2.17MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 116M/862M [00:33<04:19, 2.87MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<03:29, 3.55MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:33<02:53, 4.29MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<22:22, 554kB/s] .vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<20:27, 607kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:35<15:20, 809kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 119M/862M [00:35<11:17, 1.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:35<08:15, 1.50MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<06:19, 1.95MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 121M/862M [00:35<04:51, 2.54MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<16:55, 729kB/s] .vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<16:20, 755kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:37<12:33, 981kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:37<09:12, 1.34MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:37<06:45, 1.82MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 125M/862M [00:37<05:15, 2.34MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<12:33, 977kB/s] .vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<13:15, 925kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:39<10:22, 1.18MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<07:37, 1.61MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:39<05:48, 2.11MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:39<04:26, 2.75MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<09:27, 1.29MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:40<10:39, 1.14MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<08:24, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:41<06:29, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:41<04:58, 2.45MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:41<03:50, 3.16MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<03:06, 3.90MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<28:47, 421kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:42<24:12, 501kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<17:47, 681kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 136M/862M [00:43<12:48, 945kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:43<09:22, 1.29MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:43<06:59, 1.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<11:08, 1.08MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:44<11:49, 1.02MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<09:04, 1.33MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:45<06:41, 1.80MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 141M/862M [00:45<05:07, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:45<03:59, 3.00MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:46<09:23, 1.28MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<10:20, 1.16MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<08:14, 1.45MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:47<06:06, 1.96MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:47<04:43, 2.53MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:47<03:36, 3.31MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<13:46, 865kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<13:40, 872kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<10:22, 1.15MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:48<07:52, 1.51MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:49<05:46, 2.06MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:49<04:34, 2.60MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:50<08:32, 1.39MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<09:58, 1.19MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<07:57, 1.49MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<05:53, 2.00MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:51<04:36, 2.56MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<03:30, 3.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<15:45, 748kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:52<14:55, 790kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<11:25, 1.03MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<08:23, 1.40MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:53<06:15, 1.88MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 158M/862M [00:53<04:44, 2.47MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<09:35, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:54<10:39, 1.10MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<08:16, 1.41MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<06:19, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:55<04:43, 2.47MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:55<03:46, 3.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<08:13, 1.42MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:56<09:26, 1.23MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:56<07:34, 1.54MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<05:38, 2.06MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 166M/862M [00:57<04:19, 2.69MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<08:23, 1.38MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [00:58<09:30, 1.22MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:58<07:34, 1.53MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:40, 2.04MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [00:59<04:19, 2.67MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:59<03:22, 3.42MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:00<21:08, 545kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<21:59, 523kB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:00<17:16, 666kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:00<12:30, 919kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:01<09:12, 1.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:01<06:47, 1.69MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:01<05:05, 2.25MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:02<39:20, 291kB/s] .vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<34:54, 328kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<26:14, 436kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:03<18:59, 602kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<13:38, 837kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:03<09:53, 1.15MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<07:13, 1.57MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:05<56:28, 201kB/s] .vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<50:53, 223kB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:06<38:20, 297kB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:06<27:28, 413kB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:06<19:35, 579kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:06<13:59, 808kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<16:52, 670kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:07<13:29, 838kB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:07<09:53, 1.14MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:08<07:10, 1.57MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:08<05:25, 2.07MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<11:11, 1.00MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<14:52, 755kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:09<11:47, 952kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:10<09:01, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:10<06:35, 1.70MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:10<05:01, 2.22MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:10<03:48, 2.94MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<27:04, 413kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:11<22:02, 506kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:11<16:11, 689kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:12<11:38, 956kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:12<08:21, 1.33MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<14:51, 747kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:13<16:40, 666kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:14<13:08, 844kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:14<09:31, 1.16MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:14<07:01, 1.57MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:14<05:07, 2.15MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:15<17:56, 615kB/s] .vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<15:23, 717kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:15<11:18, 974kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:16<08:19, 1.32MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:16<06:03, 1.81MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:17<09:11, 1.19MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 205M/862M [01:18<17:50, 614kB/s] .vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<15:07, 724kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:18<11:07, 983kB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:18<08:12, 1.33MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:18<05:58, 1.83MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:19<08:03, 1.35MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<07:28, 1.46MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:20<06:36, 1.65MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:20<04:59, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:20<03:44, 2.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:21<05:58, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<08:40, 1.25MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<07:13, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:22<05:22, 2.01MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:22<04:00, 2.68MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<07:41, 1.40MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<16:15, 661kB/s] .vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:24<13:59, 768kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:24<10:17, 1.04MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:24<07:34, 1.42MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:24<05:29, 1.95MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<09:18, 1.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:26<08:23, 1.27MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:26<06:19, 1.68MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:26<04:36, 2.31MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<07:18, 1.45MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:28<08:56, 1.19MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:28<07:04, 1.50MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:28<05:13, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:28<03:50, 2.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<11:41, 902kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:30<18:53, 558kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<15:25, 683kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<11:30, 915kB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:31<08:21, 1.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:31<06:00, 1.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<10:39, 983kB/s] .vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:32<10:27, 1.00MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:33<08:01, 1.31MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<06:01, 1.73MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:33<04:26, 2.35MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<06:26, 1.62MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<07:34, 1.37MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<05:57, 1.74MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:35<04:31, 2.30MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:35<03:20, 3.09MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<08:31, 1.21MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<13:28, 767kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<11:21, 910kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<08:21, 1.24MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:37<06:02, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<07:16, 1.41MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<07:27, 1.37MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:39<05:49, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:39<04:11, 2.44MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<09:37, 1.06MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<11:31, 885kB/s] .vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<08:59, 1.13MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<06:55, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:41<04:59, 2.03MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 254M/862M [01:43<06:56, 1.46MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<07:19, 1.38MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<05:49, 1.74MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<04:22, 2.31MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:43<03:13, 3.12MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:46<10:09, 991kB/s] .vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:46<17:23, 578kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:46<14:32, 692kB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:46<10:42, 938kB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<07:40, 1.31MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:48<08:13, 1.21MB/s].vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:48<07:38, 1.31MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<05:46, 1.73MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:48<04:13, 2.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:50<06:17, 1.58MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:50<06:12, 1.60MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:50<04:46, 2.07MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<03:28, 2.84MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:52<08:46, 1.12MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:52<11:09, 883kB/s] .vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:52<08:59, 1.10MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:52<06:35, 1.49MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<04:44, 2.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:54<10:50, 903kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:54<09:08, 1.07MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:54<06:47, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<04:56, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:56<06:36, 1.47MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:56<06:45, 1.44MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:56<05:14, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:56<03:47, 2.55MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:58<39:04, 247kB/s] .vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:58<28:45, 335kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:58<20:24, 472kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:58<14:23, 667kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [02:00<15:37, 613kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [02:00<12:47, 749kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 289M/862M [02:00<09:24, 1.02MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<06:40, 1.42MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<11:10, 851kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:02<09:40, 983kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [02:02<07:12, 1.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:02<05:06, 1.85MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:04<31:36, 299kB/s] .vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:04<23:59, 393kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:04<17:13, 547kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:06<13:41, 684kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:06<13:13, 708kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:06<10:09, 922kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 302M/862M [02:06<07:18, 1.28MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:08<06:59, 1.33MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:08<06:45, 1.37MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:08<05:11, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<03:44, 2.47MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:10<07:05, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:10<06:45, 1.36MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:10<05:04, 1.82MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:10<03:48, 2.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:12<05:04, 1.80MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:12<07:03, 1.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:12<05:52, 1.56MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:12<04:17, 2.12MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:14<05:06, 1.78MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<05:18, 1.71MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:14<04:09, 2.18MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:16<04:41, 1.92MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:16<10:55, 827kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:16<09:28, 952kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:17<07:01, 1.28MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:17<05:05, 1.77MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:18<05:55, 1.51MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:18<05:52, 1.53MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:18<04:29, 1.99MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:19<03:18, 2.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:20<05:15, 1.69MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:20<07:22, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 329M/862M [02:21<05:57, 1.49MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:21<04:24, 2.01MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:21<03:13, 2.74MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:22<07:42, 1.15MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:22<07:05, 1.24MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:23<05:23, 1.63MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:23<03:52, 2.27MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:24<08:18, 1.05MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:24<07:33, 1.16MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:25<05:42, 1.53MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:25<04:05, 2.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:27<10:37, 818kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:27<11:02, 787kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:27<08:36, 1.01MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:27<06:12, 1.39MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:27<04:27, 1.93MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:29<19:09, 450kB/s] .vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:29<14:36, 590kB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:29<10:42, 804kB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:29<07:37, 1.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:31<07:48, 1.09MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:31<07:15, 1.18MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:31<05:23, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:31<04:01, 2.12MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:33<04:57, 1.71MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:33<05:13, 1.62MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:33<04:06, 2.06MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:33<03:00, 2.81MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:35<05:02, 1.67MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:35<05:00, 1.68MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:35<04:27, 1.89MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:35<03:20, 2.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:35<02:27, 3.39MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:37<08:02, 1.04MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:37<07:24, 1.12MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:37<05:31, 1.51MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:37<04:05, 2.03MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:39<04:45, 1.74MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:39<05:08, 1.61MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:39<04:01, 2.05MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:39<02:55, 2.81MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<07:00, 1.17MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<11:48, 695kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:41<10:03, 816kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:41<07:27, 1.10MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:42<05:18, 1.54MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<08:32, 953kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:43<07:09, 1.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:43<05:14, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 377M/862M [02:43<03:47, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<07:40, 1.05MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:45<06:54, 1.17MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:45<05:11, 1.55MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<04:59, 1.60MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:47<06:35, 1.21MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:47<05:15, 1.52MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:47<03:54, 2.04MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:49<04:12, 1.89MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:49<04:25, 1.79MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:49<03:27, 2.29MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:49<02:29, 3.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:51<10:43, 733kB/s] .vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:51<08:57, 877kB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:51<06:34, 1.19MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:51<04:44, 1.65MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:53<05:42, 1.36MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:53<05:25, 1.44MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:53<04:09, 1.87MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 399M/862M [02:53<02:58, 2.60MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:55<27:09, 284kB/s] .vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:55<20:27, 377kB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:55<14:34, 528kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:55<10:22, 740kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:57<09:11, 832kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:57<07:54, 968kB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:57<05:53, 1.30MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:57<04:13, 1.80MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:59<06:22, 1.19MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:59<05:34, 1.36MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:59<04:42, 1.61MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:59<03:28, 2.18MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:59<02:30, 2.99MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<1:54:01, 65.9kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [03:01<1:21:13, 92.5kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [03:01<57:05, 131kB/s]   .vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:01<39:50, 187kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:03<33:42, 221kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<25:01, 297kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:03<17:48, 417kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:03<12:29, 592kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:05<14:12, 519kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:05<11:06, 664kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 420M/862M [03:05<08:28, 870kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:05<06:03, 1.21MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:07<05:54, 1.24MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:07<05:27, 1.34MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:07<04:06, 1.77MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:07<02:57, 2.46MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:09<08:16, 875kB/s] .vector_cache/glove.6B.zip:  50%|████▉     | 428M/862M [03:09<07:08, 1.01MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:09<05:19, 1.36MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:09<03:47, 1.90MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:11<11:11, 641kB/s] .vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:11<09:09, 783kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:11<06:41, 1.07MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:11<04:48, 1.48MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:13<05:46, 1.23MB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:13<05:22, 1.32MB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:13<04:05, 1.73MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:13<02:55, 2.41MB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:15<11:01, 639kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:15<10:26, 674kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:15<07:53, 891kB/s].vector_cache/glove.6B.zip:  51%|█████     | 442M/862M [03:15<05:41, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:15<04:02, 1.72MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:17<1:40:43, 69.2kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 444M/862M [03:17<1:11:48, 97.0kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:17<50:26, 138kB/s]   .vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:17<35:13, 196kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:19<28:16, 244kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<20:47, 332kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:19<15:16, 451kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:19<10:46, 637kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:21<09:08, 747kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:21<07:43, 883kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:21<05:42, 1.19MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:21<04:03, 1.67MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:23<09:59, 676kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:23<07:41, 879kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:23<05:53, 1.15MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:23<04:11, 1.60MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:25<05:28, 1.22MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 461M/862M [03:25<04:41, 1.42MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:25<03:29, 1.91MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:27<03:44, 1.77MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 465M/862M [03:27<03:46, 1.75MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:27<02:53, 2.28MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:27<02:07, 3.09MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:29<04:08, 1.58MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:29<04:02, 1.62MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:29<03:03, 2.14MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:29<02:13, 2.91MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:31<06:12, 1.05MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:32<10:50, 598kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:32<09:08, 709kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:32<06:44, 960kB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:32<04:51, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:33<04:49, 1.33MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:34<03:56, 1.63MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:34<02:51, 2.23MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:35<03:37, 1.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:36<04:49, 1.31MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:36<04:01, 1.57MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:36<03:15, 1.94MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:36<02:21, 2.66MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:36<01:50, 3.42MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:37<33:27, 188kB/s] .vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:38<26:52, 234kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:38<19:41, 319kB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:38<13:55, 449kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:38<09:55, 629kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:38<07:02, 882kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:39<27:21, 227kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:40<22:38, 274kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 490M/862M [03:40<16:39, 372kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:40<11:48, 523kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:40<08:22, 735kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<08:01, 764kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:41<08:42, 704kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 494M/862M [03:42<06:49, 898kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:42<04:57, 1.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:42<03:34, 1.70MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:43<05:38, 1.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:43<05:24, 1.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:44<04:07, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:44<03:01, 2.00MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<03:59, 1.50MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 502M/862M [03:45<05:31, 1.09MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:46<04:26, 1.35MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:46<03:20, 1.79MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:46<02:28, 2.40MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:46<01:53, 3.15MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:47<05:51, 1.01MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:47<06:34, 902kB/s] .vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:48<05:14, 1.13MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:48<03:48, 1.55MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:48<02:49, 2.08MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:49<04:00, 1.46MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:49<05:16, 1.11MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:50<04:18, 1.36MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:50<03:08, 1.85MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:50<02:21, 2.47MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:51<03:42, 1.56MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:51<05:04, 1.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:51<04:10, 1.39MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:52<03:12, 1.80MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:52<02:23, 2.41MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:52<01:47, 3.21MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:53<05:31, 1.04MB/s].vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:53<06:15, 916kB/s] .vector_cache/glove.6B.zip:  60%|██████    | 519M/862M [03:53<04:57, 1.15MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:54<03:46, 1.52MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:54<02:46, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:54<02:02, 2.77MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<06:32, 865kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<06:45, 836kB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:55<05:13, 1.08MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:56<03:52, 1.46MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:56<02:50, 1.97MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:56<02:05, 2.67MB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<13:32, 412kB/s] .vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<11:48, 473kB/s].vector_cache/glove.6B.zip:  61%|██████    | 527M/862M [03:57<08:50, 630kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:58<06:18, 881kB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:58<04:32, 1.22MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:59<05:13, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:59<05:55, 931kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:59<04:41, 1.18MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [04:00<03:25, 1.60MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:00<02:28, 2.21MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:01<24:08, 226kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [04:01<19:14, 283kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:01<14:01, 388kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [04:01<10:04, 540kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [04:02<07:07, 759kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:02<05:04, 1.06MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [04:03<38:35, 139kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:03<29:03, 185kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [04:03<20:47, 258kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:03<14:39, 365kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:04<10:19, 516kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:05<09:23, 565kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:05<08:37, 616kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:05<06:28, 818kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [04:05<04:41, 1.13MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:06<03:23, 1.55MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<04:10, 1.25MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<05:06, 1.02MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:07<04:06, 1.28MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:07<02:59, 1.74MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:08<02:11, 2.37MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<04:42, 1.10MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<05:25, 952kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [04:09<04:13, 1.22MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:09<03:03, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:09<02:15, 2.26MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 556M/862M [04:11<03:39, 1.39MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 556M/862M [04:11<04:32, 1.12MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:11<03:36, 1.41MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:11<02:40, 1.90MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:11<01:58, 2.56MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:13<03:24, 1.48MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 560M/862M [04:13<04:23, 1.15MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:13<03:29, 1.44MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:13<02:42, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:13<01:59, 2.51MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:14<01:29, 3.34MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:15<14:48, 335kB/s] .vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:15<12:16, 404kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:15<09:01, 549kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:15<06:27, 765kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:16<04:36, 1.07MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 568M/862M [04:17<04:57, 986kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:17<05:31, 886kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:17<04:18, 1.14MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:17<03:09, 1.54MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:18<02:18, 2.10MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:19<03:22, 1.43MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:19<04:21, 1.11MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:19<03:31, 1.37MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:19<02:33, 1.87MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:20<01:51, 2.56MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<07:05, 670kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<06:48, 698kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:21<05:13, 908kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:21<03:45, 1.26MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:22<02:54, 1.62MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:23<03:11, 1.47MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:23<04:51, 966kB/s] .vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:23<03:59, 1.17MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:23<03:06, 1.51MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:23<02:16, 2.05MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:24<01:45, 2.65MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:24<01:20, 3.45MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:26<1:01:11, 75.5kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:26<46:51, 98.6kB/s]  .vector_cache/glove.6B.zip:  68%|██████▊   | 585M/862M [04:26<33:54, 136kB/s] .vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:26<23:56, 192kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:26<16:48, 273kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:26<11:50, 386kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:28<10:15, 443kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:28<08:26, 539kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:28<06:12, 730kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:28<04:26, 1.02MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:28<03:13, 1.40MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:30<04:11, 1.07MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:30<05:29, 817kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:30<04:25, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:30<03:15, 1.37MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:30<02:23, 1.86MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:30<01:45, 2.51MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:32<05:43, 771kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:32<05:11, 850kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:32<03:51, 1.14MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:32<02:51, 1.53MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:32<02:05, 2.09MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:32<01:34, 2.76MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:34<46:54, 92.6kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:34<35:03, 124kB/s] .vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:34<25:10, 172kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:34<17:50, 243kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:34<12:53, 335kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:34<09:04, 474kB/s].vector_cache/glove.6B.zip:  70%|███████   | 605M/862M [04:34<06:27, 663kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:35<04:34, 929kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:36<13:12, 322kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:37<13:29, 315kB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:37<10:27, 406kB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:37<07:31, 563kB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:37<05:23, 782kB/s].vector_cache/glove.6B.zip:  71%|███████   | 610M/862M [04:37<03:51, 1.09MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:38<04:09, 1.01MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:38<03:57, 1.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:39<03:01, 1.38MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:39<02:13, 1.86MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:39<01:38, 2.51MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:40<03:16, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:40<04:22, 940kB/s] .vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:41<03:34, 1.15MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:41<02:37, 1.56MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:41<01:54, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:42<02:56, 1.37MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:43<04:17, 940kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:43<03:25, 1.18MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:43<02:31, 1.59MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:43<01:52, 2.13MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:43<01:24, 2.84MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:44<07:42, 516kB/s] .vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:44<07:25, 535kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:45<05:41, 697kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 625M/862M [04:45<04:03, 971kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:45<02:58, 1.32MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:45<02:09, 1.82MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:46<08:37, 453kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:46<08:00, 488kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:47<06:05, 640kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:47<04:25, 880kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 630M/862M [04:47<03:09, 1.22MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:47<02:17, 1.67MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:48<04:51, 789kB/s] .vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:48<04:22, 876kB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:49<03:16, 1.17MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 634M/862M [04:49<02:24, 1.58MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:49<01:44, 2.16MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:50<03:04, 1.23MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:50<04:03, 929kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:51<03:18, 1.14MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:51<02:25, 1.55MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 639M/862M [04:51<01:47, 2.08MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:53<02:34, 1.44MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:53<05:27, 677kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:53<04:41, 788kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:53<03:27, 1.07MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:53<02:30, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:53<01:49, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:54<03:23, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:55<04:13, 859kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:55<03:22, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:55<02:28, 1.46MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:55<01:47, 2.00MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:56<02:51, 1.24MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:57<03:37, 980kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:57<02:57, 1.20MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:57<02:10, 1.62MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:57<01:34, 2.22MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:58<02:55, 1.19MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:59<03:47, 921kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:59<03:03, 1.14MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:59<02:13, 1.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:59<01:37, 2.11MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:00<02:24, 1.42MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [05:01<02:28, 1.38MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [05:01<01:52, 1.81MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [05:01<01:27, 2.34MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [05:01<01:04, 3.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:02<02:10, 1.54MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:03<03:12, 1.05MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [05:03<02:38, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 663M/862M [05:03<01:55, 1.72MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [05:03<01:24, 2.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:04<03:10, 1.04MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [05:04<03:42, 884kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:05<02:58, 1.10MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 667M/862M [05:05<02:10, 1.50MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [05:05<01:34, 2.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:06<02:43, 1.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:06<03:23, 948kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:07<02:44, 1.17MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [05:07<02:00, 1.59MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:07<01:27, 2.16MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:08<02:25, 1.30MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:08<03:08, 1.00MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:09<02:31, 1.24MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 675M/862M [05:09<01:51, 1.68MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:09<01:20, 2.31MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:10<04:13, 727kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:10<04:14, 724kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:11<03:13, 950kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:11<02:25, 1.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:11<01:44, 1.74MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:11<01:16, 2.35MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:12<05:47, 519kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:12<05:19, 565kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:13<03:56, 760kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:13<02:55, 1.02MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:13<02:04, 1.43MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<02:40, 1.10MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<02:59, 984kB/s] .vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:14<02:19, 1.26MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:15<01:42, 1.71MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 688M/862M [05:15<01:16, 2.27MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:15<00:56, 3.05MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:16<14:17, 201kB/s] .vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:16<11:03, 259kB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:16<07:57, 360kB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:17<05:40, 502kB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:17<04:00, 705kB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:17<02:50, 986kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:18<06:11, 453kB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:18<05:29, 509kB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:18<04:07, 678kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:19<02:55, 946kB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:19<02:04, 1.32MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<04:36, 593kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:20<04:07, 661kB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:20<03:06, 875kB/s].vector_cache/glove.6B.zip:  81%|████████  | 701M/862M [05:21<02:12, 1.22MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:21<01:35, 1.68MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:22<03:17, 810kB/s] .vector_cache/glove.6B.zip:  81%|████████▏ | 703M/862M [05:22<03:09, 844kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:22<02:22, 1.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 704M/862M [05:23<01:45, 1.50MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:23<01:15, 2.08MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:24<02:43, 954kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:24<02:39, 976kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:24<02:02, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 709M/862M [05:25<01:27, 1.75MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:26<01:49, 1.38MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:26<01:58, 1.27MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:26<01:32, 1.64MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:26<01:07, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:27<00:49, 2.99MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<03:38, 675kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<03:25, 717kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:28<02:35, 946kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:28<01:52, 1.29MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:29<01:19, 1.81MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:30<04:56, 482kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:30<04:07, 577kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:30<03:01, 787kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:30<02:09, 1.09MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:32<01:58, 1.17MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:32<02:03, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:32<01:34, 1.46MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 725M/862M [05:32<01:09, 1.96MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:34<01:15, 1.78MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:34<01:18, 1.73MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:34<01:11, 1.88MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:34<00:53, 2.50MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:34<00:39, 3.34MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:36<01:48, 1.20MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:37<02:23, 912kB/s] .vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:37<01:56, 1.12MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:37<01:24, 1.53MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:38<01:22, 1.54MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:38<01:34, 1.33MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:39<01:14, 1.69MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 737M/862M [05:39<00:55, 2.26MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:39<00:39, 3.09MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:40<05:16, 386kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:41<04:15, 478kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:41<03:05, 655kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:41<02:11, 913kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:41<01:32, 1.28MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:42<05:29, 359kB/s] .vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:43<04:12, 467kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 745M/862M [05:43<03:00, 647kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:43<02:06, 912kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:44<02:24, 792kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:45<02:01, 940kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:45<01:29, 1.26MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:45<01:02, 1.77MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:46<07:50, 234kB/s] .vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:46<05:47, 316kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:47<04:04, 444kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:47<02:49, 628kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:48<02:50, 619kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:48<02:23, 735kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:49<01:45, 998kB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:49<01:15, 1.38MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:50<01:15, 1.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:50<01:15, 1.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:51<00:57, 1.74MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 763M/862M [05:51<00:41, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:52<01:01, 1.59MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:52<00:57, 1.69MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:53<00:43, 2.22MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:53<00:31, 3.02MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:54<01:07, 1.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:54<01:00, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:55<00:45, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:55<00:32, 2.80MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:56<02:23, 621kB/s] .vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:56<01:57, 759kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:56<01:25, 1.03MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:58<01:12, 1.18MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:58<01:06, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 778M/862M [05:58<00:50, 1.68MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:00<00:47, 1.71MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [06:00<00:47, 1.69MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [06:00<00:36, 2.17MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [06:02<00:37, 2.04MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:02<00:40, 1.88MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [06:02<00:31, 2.38MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [06:04<00:34, 2.14MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:04<00:52, 1.40MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [06:04<00:42, 1.71MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [06:04<00:30, 2.31MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:05<00:21, 3.15MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:06<02:04, 550kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:06<01:40, 683kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [06:06<01:12, 932kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:08<00:59, 1.09MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:08<00:53, 1.20MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:08<00:39, 1.59MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:10<00:36, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:10<00:36, 1.64MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:10<00:28, 2.11MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:12<00:28, 2.01MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:12<00:30, 1.85MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:12<00:23, 2.39MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:12<00:16, 3.25MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:14<00:36, 1.44MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:14<00:35, 1.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:14<00:26, 1.92MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:16<00:25, 1.88MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:16<00:26, 1.78MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:16<00:20, 2.30MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:16<00:14, 3.08MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:18<00:23, 1.88MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:18<00:24, 1.77MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 819M/862M [06:18<00:18, 2.29MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:18<00:13, 3.10MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:20<00:23, 1.70MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:20<00:23, 1.65MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 823M/862M [06:20<00:18, 2.15MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 825M/862M [06:20<00:12, 2.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:22<00:24, 1.46MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:22<00:23, 1.49MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:22<00:17, 1.93MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:22<00:11, 2.68MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:24<10:19, 50.8kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:24<07:17, 71.6kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:24<04:59, 102kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:24<03:19, 145kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:26<02:20, 195kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:26<01:42, 264kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:26<01:11, 371kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:28<00:47, 487kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:28<00:37, 614kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:28<00:26, 850kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:28<00:17, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:30<00:16, 1.17MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:30<00:14, 1.28MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 844M/862M [06:30<00:10, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:32<00:08, 1.71MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:32<00:08, 1.68MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:32<00:06, 2.15MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:32<00:03, 2.97MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:34<00:13, 807kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 851M/862M [06:34<00:11, 940kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:34<00:07, 1.28MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:34<00:05, 1.74MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:36<00:04, 1.53MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:36<00:04, 1.54MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:36<00:02, 2.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:36<00:01, 2.70MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:38<00:01, 1.85MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:38<00:01, 1.75MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:38<00:00, 2.24MB/s].vector_cache/glove.6B.zip: 862MB [06:38, 2.16MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<27:10:24,  4.09it/s]  0%|          | 814/400000 [00:00<18:59:12,  5.84it/s]  0%|          | 1641/400000 [00:00<13:16:02,  8.34it/s]  1%|          | 2445/400000 [00:00<9:16:21, 11.91it/s]   1%|          | 3259/400000 [00:00<6:28:53, 17.00it/s]  1%|          | 4069/400000 [00:00<4:31:54, 24.27it/s]  1%|          | 4892/400000 [00:00<3:10:11, 34.63it/s]  1%|▏         | 5715/400000 [00:00<2:13:05, 49.38it/s]  2%|▏         | 6469/400000 [00:01<1:33:14, 70.34it/s]  2%|▏         | 7230/400000 [00:01<1:05:24, 100.09it/s]  2%|▏         | 8035/400000 [00:01<45:55, 142.22it/s]    2%|▏         | 8804/400000 [00:01<32:21, 201.51it/s]  2%|▏         | 9628/400000 [00:01<22:50, 284.88it/s]  3%|▎         | 10462/400000 [00:01<16:11, 401.09it/s]  3%|▎         | 11305/400000 [00:01<11:32, 561.52it/s]  3%|▎         | 12114/400000 [00:01<08:18, 778.62it/s]  3%|▎         | 12933/400000 [00:01<06:02, 1068.73it/s]  3%|▎         | 13742/400000 [00:01<04:27, 1441.69it/s]  4%|▎         | 14570/400000 [00:02<03:21, 1916.54it/s]  4%|▍         | 15401/400000 [00:02<02:34, 2491.45it/s]  4%|▍         | 16234/400000 [00:02<02:01, 3154.57it/s]  4%|▍         | 17054/400000 [00:02<01:39, 3854.78it/s]  4%|▍         | 17877/400000 [00:02<01:23, 4585.84it/s]  5%|▍         | 18713/400000 [00:02<01:11, 5303.08it/s]  5%|▍         | 19553/400000 [00:02<01:03, 5962.50it/s]  5%|▌         | 20395/400000 [00:02<00:58, 6533.43it/s]  5%|▌         | 21227/400000 [00:02<00:54, 6920.44it/s]  6%|▌         | 22058/400000 [00:02<00:51, 7284.40it/s]  6%|▌         | 22895/400000 [00:03<00:49, 7578.92it/s]  6%|▌         | 23733/400000 [00:03<00:48, 7800.24it/s]  6%|▌         | 24565/400000 [00:03<00:47, 7928.79it/s]  6%|▋         | 25395/400000 [00:03<00:46, 8021.40it/s]  7%|▋         | 26232/400000 [00:03<00:46, 8121.83it/s]  7%|▋         | 27069/400000 [00:03<00:45, 8194.41it/s]  7%|▋         | 27902/400000 [00:03<00:45, 8217.21it/s]  7%|▋         | 28736/400000 [00:03<00:44, 8251.14it/s]  7%|▋         | 29568/400000 [00:03<00:44, 8249.43it/s]  8%|▊         | 30400/400000 [00:03<00:44, 8269.51it/s]  8%|▊         | 31237/400000 [00:04<00:44, 8297.45it/s]  8%|▊         | 32082/400000 [00:04<00:44, 8341.02it/s]  8%|▊         | 32920/400000 [00:04<00:43, 8351.03it/s]  8%|▊         | 33760/400000 [00:04<00:43, 8364.26it/s]  9%|▊         | 34598/400000 [00:04<00:43, 8327.67it/s]  9%|▉         | 35442/400000 [00:04<00:43, 8359.01it/s]  9%|▉         | 36279/400000 [00:04<00:43, 8358.37it/s]  9%|▉         | 37116/400000 [00:04<00:43, 8328.25it/s]  9%|▉         | 37956/400000 [00:04<00:43, 8347.69it/s] 10%|▉         | 38791/400000 [00:04<00:43, 8294.31it/s] 10%|▉         | 39633/400000 [00:05<00:43, 8330.30it/s] 10%|█         | 40467/400000 [00:05<00:43, 8308.56it/s] 10%|█         | 41303/400000 [00:05<00:43, 8323.14it/s] 11%|█         | 42136/400000 [00:05<00:43, 8135.76it/s] 11%|█         | 42951/400000 [00:05<00:43, 8124.17it/s] 11%|█         | 43798/400000 [00:05<00:43, 8224.08it/s] 11%|█         | 44642/400000 [00:05<00:42, 8285.84it/s] 11%|█▏        | 45485/400000 [00:05<00:42, 8325.76it/s] 12%|█▏        | 46324/400000 [00:05<00:42, 8344.30it/s] 12%|█▏        | 47159/400000 [00:05<00:42, 8345.11it/s] 12%|█▏        | 48002/400000 [00:06<00:42, 8367.57it/s] 12%|█▏        | 48852/400000 [00:06<00:41, 8404.03it/s] 12%|█▏        | 49703/400000 [00:06<00:41, 8434.30it/s] 13%|█▎        | 50547/400000 [00:06<00:42, 8209.08it/s] 13%|█▎        | 51370/400000 [00:06<00:42, 8186.70it/s] 13%|█▎        | 52210/400000 [00:06<00:42, 8247.94it/s] 13%|█▎        | 53059/400000 [00:06<00:41, 8317.69it/s] 13%|█▎        | 53905/400000 [00:06<00:41, 8358.06it/s] 14%|█▎        | 54742/400000 [00:06<00:41, 8356.02it/s] 14%|█▍        | 55578/400000 [00:06<00:41, 8341.10it/s] 14%|█▍        | 56413/400000 [00:07<00:41, 8299.62it/s] 14%|█▍        | 57249/400000 [00:07<00:41, 8317.25it/s] 15%|█▍        | 58098/400000 [00:07<00:40, 8367.86it/s] 15%|█▍        | 58935/400000 [00:07<00:40, 8331.97it/s] 15%|█▍        | 59783/400000 [00:07<00:40, 8374.29it/s] 15%|█▌        | 60621/400000 [00:07<00:40, 8357.67it/s] 15%|█▌        | 61466/400000 [00:07<00:40, 8384.42it/s] 16%|█▌        | 62305/400000 [00:07<00:40, 8376.85it/s] 16%|█▌        | 63143/400000 [00:07<00:40, 8343.40it/s] 16%|█▌        | 63985/400000 [00:07<00:40, 8365.87it/s] 16%|█▌        | 64822/400000 [00:08<00:40, 8228.19it/s] 16%|█▋        | 65669/400000 [00:08<00:40, 8297.94it/s] 17%|█▋        | 66516/400000 [00:08<00:39, 8347.49it/s] 17%|█▋        | 67352/400000 [00:08<00:40, 8281.21it/s] 17%|█▋        | 68187/400000 [00:08<00:39, 8301.11it/s] 17%|█▋        | 69018/400000 [00:08<00:40, 8199.09it/s] 17%|█▋        | 69852/400000 [00:08<00:40, 8238.19it/s] 18%|█▊        | 70677/400000 [00:08<00:40, 8101.20it/s] 18%|█▊        | 71493/400000 [00:08<00:40, 8117.58it/s] 18%|█▊        | 72326/400000 [00:09<00:40, 8177.93it/s] 18%|█▊        | 73145/400000 [00:09<00:39, 8180.97it/s] 18%|█▊        | 73978/400000 [00:09<00:39, 8224.52it/s] 19%|█▊        | 74803/400000 [00:09<00:39, 8229.02it/s] 19%|█▉        | 75627/400000 [00:09<00:40, 8087.40it/s] 19%|█▉        | 76441/400000 [00:09<00:39, 8102.66it/s] 19%|█▉        | 77268/400000 [00:09<00:39, 8149.39it/s] 20%|█▉        | 78102/400000 [00:09<00:39, 8205.16it/s] 20%|█▉        | 78929/400000 [00:09<00:39, 8224.09it/s] 20%|█▉        | 79752/400000 [00:09<00:39, 8195.42it/s] 20%|██        | 80584/400000 [00:10<00:38, 8230.53it/s] 20%|██        | 81408/400000 [00:10<00:38, 8206.66it/s] 21%|██        | 82229/400000 [00:10<00:38, 8190.50it/s] 21%|██        | 83065/400000 [00:10<00:38, 8240.40it/s] 21%|██        | 83897/400000 [00:10<00:38, 8261.97it/s] 21%|██        | 84724/400000 [00:10<00:38, 8230.82it/s] 21%|██▏       | 85563/400000 [00:10<00:37, 8277.50it/s] 22%|██▏       | 86391/400000 [00:10<00:38, 8221.29it/s] 22%|██▏       | 87239/400000 [00:10<00:37, 8295.24it/s] 22%|██▏       | 88069/400000 [00:10<00:37, 8279.98it/s] 22%|██▏       | 88910/400000 [00:11<00:37, 8317.42it/s] 22%|██▏       | 89742/400000 [00:11<00:37, 8257.74it/s] 23%|██▎       | 90576/400000 [00:11<00:37, 8280.57it/s] 23%|██▎       | 91405/400000 [00:11<00:37, 8227.66it/s] 23%|██▎       | 92228/400000 [00:11<00:37, 8118.62it/s] 23%|██▎       | 93075/400000 [00:11<00:37, 8220.33it/s] 23%|██▎       | 93907/400000 [00:11<00:37, 8247.40it/s] 24%|██▎       | 94749/400000 [00:11<00:36, 8297.97it/s] 24%|██▍       | 95580/400000 [00:11<00:36, 8247.47it/s] 24%|██▍       | 96417/400000 [00:11<00:36, 8282.84it/s] 24%|██▍       | 97264/400000 [00:12<00:36, 8336.43it/s] 25%|██▍       | 98105/400000 [00:12<00:36, 8356.39it/s] 25%|██▍       | 98941/400000 [00:12<00:36, 8258.53it/s] 25%|██▍       | 99784/400000 [00:12<00:36, 8308.31it/s] 25%|██▌       | 100628/400000 [00:12<00:35, 8346.59it/s] 25%|██▌       | 101463/400000 [00:12<00:35, 8341.97it/s] 26%|██▌       | 102298/400000 [00:12<00:35, 8337.83it/s] 26%|██▌       | 103134/400000 [00:12<00:35, 8341.57it/s] 26%|██▌       | 103971/400000 [00:12<00:35, 8347.64it/s] 26%|██▌       | 104807/400000 [00:12<00:35, 8349.79it/s] 26%|██▋       | 105643/400000 [00:13<00:35, 8289.99it/s] 27%|██▋       | 106476/400000 [00:13<00:35, 8300.37it/s] 27%|██▋       | 107310/400000 [00:13<00:35, 8311.72it/s] 27%|██▋       | 108142/400000 [00:13<00:35, 8212.42it/s] 27%|██▋       | 108979/400000 [00:13<00:35, 8257.94it/s] 27%|██▋       | 109814/400000 [00:13<00:35, 8283.84it/s] 28%|██▊       | 110643/400000 [00:13<00:34, 8272.82it/s] 28%|██▊       | 111471/400000 [00:13<00:35, 8213.93it/s] 28%|██▊       | 112293/400000 [00:13<00:35, 8171.93it/s] 28%|██▊       | 113111/400000 [00:13<00:35, 8154.45it/s] 28%|██▊       | 113937/400000 [00:14<00:34, 8185.74it/s] 29%|██▊       | 114774/400000 [00:14<00:34, 8239.17it/s] 29%|██▉       | 115613/400000 [00:14<00:34, 8283.00it/s] 29%|██▉       | 116442/400000 [00:14<00:34, 8258.30it/s] 29%|██▉       | 117283/400000 [00:14<00:34, 8301.31it/s] 30%|██▉       | 118125/400000 [00:14<00:33, 8334.65it/s] 30%|██▉       | 118959/400000 [00:14<00:33, 8312.32it/s] 30%|██▉       | 119796/400000 [00:14<00:33, 8327.90it/s] 30%|███       | 120629/400000 [00:14<00:33, 8308.41it/s] 30%|███       | 121460/400000 [00:14<00:33, 8307.88it/s] 31%|███       | 122299/400000 [00:15<00:33, 8330.10it/s] 31%|███       | 123144/400000 [00:15<00:33, 8363.86it/s] 31%|███       | 123981/400000 [00:15<00:33, 8363.44it/s] 31%|███       | 124818/400000 [00:15<00:32, 8342.62it/s] 31%|███▏      | 125655/400000 [00:15<00:32, 8349.20it/s] 32%|███▏      | 126502/400000 [00:15<00:32, 8382.38it/s] 32%|███▏      | 127348/400000 [00:15<00:32, 8402.63it/s] 32%|███▏      | 128189/400000 [00:15<00:32, 8368.50it/s] 32%|███▏      | 129030/400000 [00:15<00:32, 8378.90it/s] 32%|███▏      | 129868/400000 [00:15<00:32, 8352.21it/s] 33%|███▎      | 130711/400000 [00:16<00:32, 8374.27it/s] 33%|███▎      | 131549/400000 [00:16<00:32, 8326.05it/s] 33%|███▎      | 132389/400000 [00:16<00:32, 8345.81it/s] 33%|███▎      | 133229/400000 [00:16<00:31, 8359.10it/s] 34%|███▎      | 134065/400000 [00:16<00:32, 8174.20it/s] 34%|███▎      | 134911/400000 [00:16<00:32, 8257.08it/s] 34%|███▍      | 135749/400000 [00:16<00:31, 8291.32it/s] 34%|███▍      | 136597/400000 [00:16<00:31, 8344.42it/s] 34%|███▍      | 137439/400000 [00:16<00:31, 8365.44it/s] 35%|███▍      | 138276/400000 [00:16<00:32, 8157.02it/s] 35%|███▍      | 139121/400000 [00:17<00:31, 8241.71it/s] 35%|███▍      | 139966/400000 [00:17<00:31, 8302.95it/s] 35%|███▌      | 140817/400000 [00:17<00:30, 8363.35it/s] 35%|███▌      | 141655/400000 [00:17<00:31, 8267.57it/s] 36%|███▌      | 142483/400000 [00:17<00:31, 8086.13it/s] 36%|███▌      | 143331/400000 [00:17<00:31, 8199.98it/s] 36%|███▌      | 144183/400000 [00:17<00:30, 8291.91it/s] 36%|███▋      | 145032/400000 [00:17<00:30, 8347.60it/s] 36%|███▋      | 145881/400000 [00:17<00:30, 8387.79it/s] 37%|███▋      | 146721/400000 [00:18<00:30, 8377.39it/s] 37%|███▋      | 147565/400000 [00:18<00:30, 8394.03it/s] 37%|███▋      | 148414/400000 [00:18<00:29, 8420.08it/s] 37%|███▋      | 149263/400000 [00:18<00:29, 8438.91it/s] 38%|███▊      | 150111/400000 [00:18<00:29, 8449.15it/s] 38%|███▊      | 150957/400000 [00:18<00:29, 8386.79it/s] 38%|███▊      | 151805/400000 [00:18<00:29, 8411.68it/s] 38%|███▊      | 152649/400000 [00:18<00:29, 8419.47it/s] 38%|███▊      | 153499/400000 [00:18<00:29, 8440.48it/s] 39%|███▊      | 154344/400000 [00:18<00:29, 8435.63it/s] 39%|███▉      | 155188/400000 [00:19<00:29, 8400.90it/s] 39%|███▉      | 156029/400000 [00:19<00:29, 8236.63it/s] 39%|███▉      | 156871/400000 [00:19<00:29, 8289.48it/s] 39%|███▉      | 157701/400000 [00:19<00:29, 8282.15it/s] 40%|███▉      | 158549/400000 [00:19<00:28, 8338.75it/s] 40%|███▉      | 159384/400000 [00:19<00:29, 8255.14it/s] 40%|████      | 160210/400000 [00:19<00:29, 8237.83it/s] 40%|████      | 161035/400000 [00:19<00:29, 8023.00it/s] 40%|████      | 161878/400000 [00:19<00:29, 8138.97it/s] 41%|████      | 162723/400000 [00:19<00:28, 8228.42it/s] 41%|████      | 163548/400000 [00:20<00:29, 8104.32it/s] 41%|████      | 164394/400000 [00:20<00:28, 8206.88it/s] 41%|████▏     | 165226/400000 [00:20<00:28, 8239.66it/s] 42%|████▏     | 166070/400000 [00:20<00:28, 8297.40it/s] 42%|████▏     | 166918/400000 [00:20<00:27, 8348.98it/s] 42%|████▏     | 167754/400000 [00:20<00:27, 8322.42it/s] 42%|████▏     | 168600/400000 [00:20<00:27, 8362.69it/s] 42%|████▏     | 169437/400000 [00:20<00:27, 8345.39it/s] 43%|████▎     | 170272/400000 [00:20<00:27, 8294.95it/s] 43%|████▎     | 171113/400000 [00:20<00:27, 8328.76it/s] 43%|████▎     | 171947/400000 [00:21<00:27, 8284.67it/s] 43%|████▎     | 172788/400000 [00:21<00:27, 8319.78it/s] 43%|████▎     | 173634/400000 [00:21<00:27, 8358.87it/s] 44%|████▎     | 174479/400000 [00:21<00:26, 8383.97it/s] 44%|████▍     | 175318/400000 [00:21<00:26, 8357.69it/s] 44%|████▍     | 176154/400000 [00:21<00:28, 7778.84it/s] 44%|████▍     | 176987/400000 [00:21<00:28, 7935.08it/s] 44%|████▍     | 177836/400000 [00:21<00:27, 8091.62it/s] 45%|████▍     | 178685/400000 [00:21<00:26, 8205.79it/s] 45%|████▍     | 179533/400000 [00:21<00:26, 8285.70it/s] 45%|████▌     | 180375/400000 [00:22<00:26, 8325.06it/s] 45%|████▌     | 181210/400000 [00:22<00:26, 8308.42it/s] 46%|████▌     | 182056/400000 [00:22<00:26, 8352.30it/s] 46%|████▌     | 182901/400000 [00:22<00:25, 8378.84it/s] 46%|████▌     | 183744/400000 [00:22<00:25, 8393.64it/s] 46%|████▌     | 184586/400000 [00:22<00:25, 8398.98it/s] 46%|████▋     | 185427/400000 [00:22<00:25, 8378.63it/s] 47%|████▋     | 186273/400000 [00:22<00:25, 8402.68it/s] 47%|████▋     | 187123/400000 [00:22<00:25, 8428.77it/s] 47%|████▋     | 187971/400000 [00:22<00:25, 8442.58it/s] 47%|████▋     | 188817/400000 [00:23<00:25, 8446.13it/s] 47%|████▋     | 189662/400000 [00:23<00:25, 8374.84it/s] 48%|████▊     | 190512/400000 [00:23<00:24, 8410.58it/s] 48%|████▊     | 191362/400000 [00:23<00:24, 8435.15it/s] 48%|████▊     | 192206/400000 [00:23<00:24, 8338.74it/s] 48%|████▊     | 193047/400000 [00:23<00:24, 8357.33it/s] 48%|████▊     | 193883/400000 [00:23<00:24, 8327.04it/s] 49%|████▊     | 194729/400000 [00:23<00:24, 8363.90it/s] 49%|████▉     | 195576/400000 [00:23<00:24, 8392.80it/s] 49%|████▉     | 196421/400000 [00:23<00:24, 8408.50it/s] 49%|████▉     | 197262/400000 [00:24<00:24, 8374.96it/s] 50%|████▉     | 198100/400000 [00:24<00:24, 8194.38it/s] 50%|████▉     | 198921/400000 [00:24<00:25, 7940.07it/s] 50%|████▉     | 199763/400000 [00:24<00:24, 8077.94it/s] 50%|█████     | 200598/400000 [00:24<00:24, 8155.24it/s] 50%|█████     | 201432/400000 [00:24<00:24, 8207.16it/s] 51%|█████     | 202265/400000 [00:24<00:23, 8243.57it/s] 51%|█████     | 203108/400000 [00:24<00:23, 8296.19it/s] 51%|█████     | 203954/400000 [00:24<00:23, 8343.16it/s] 51%|█████     | 204791/400000 [00:24<00:23, 8350.44it/s] 51%|█████▏    | 205629/400000 [00:25<00:23, 8356.90it/s] 52%|█████▏    | 206465/400000 [00:25<00:23, 8177.41it/s] 52%|█████▏    | 207304/400000 [00:25<00:23, 8238.19it/s] 52%|█████▏    | 208145/400000 [00:25<00:23, 8287.04it/s] 52%|█████▏    | 208991/400000 [00:25<00:22, 8336.13it/s] 52%|█████▏    | 209827/400000 [00:25<00:22, 8342.25it/s] 53%|█████▎    | 210662/400000 [00:25<00:22, 8296.90it/s] 53%|█████▎    | 211493/400000 [00:25<00:22, 8246.72it/s] 53%|█████▎    | 212318/400000 [00:25<00:22, 8242.08it/s] 53%|█████▎    | 213143/400000 [00:26<00:22, 8223.77it/s] 53%|█████▎    | 213979/400000 [00:26<00:22, 8261.78it/s] 54%|█████▎    | 214809/400000 [00:26<00:22, 8271.28it/s] 54%|█████▍    | 215648/400000 [00:26<00:22, 8303.71it/s] 54%|█████▍    | 216492/400000 [00:26<00:21, 8343.61it/s] 54%|█████▍    | 217341/400000 [00:26<00:21, 8386.98it/s] 55%|█████▍    | 218180/400000 [00:26<00:22, 8249.48it/s] 55%|█████▍    | 219020/400000 [00:26<00:21, 8292.86it/s] 55%|█████▍    | 219868/400000 [00:26<00:21, 8345.27it/s] 55%|█████▌    | 220717/400000 [00:26<00:21, 8386.18it/s] 55%|█████▌    | 221556/400000 [00:27<00:21, 8384.36it/s] 56%|█████▌    | 222400/400000 [00:27<00:21, 8399.42it/s] 56%|█████▌    | 223246/400000 [00:27<00:20, 8417.04it/s] 56%|█████▌    | 224088/400000 [00:27<00:20, 8401.03it/s] 56%|█████▌    | 224936/400000 [00:27<00:20, 8424.34it/s] 56%|█████▋    | 225779/400000 [00:27<00:20, 8391.30it/s] 57%|█████▋    | 226625/400000 [00:27<00:20, 8410.23it/s] 57%|█████▋    | 227467/400000 [00:27<00:20, 8412.84it/s] 57%|█████▋    | 228309/400000 [00:27<00:20, 8405.85it/s] 57%|█████▋    | 229158/400000 [00:27<00:20, 8427.36it/s] 58%|█████▊    | 230006/400000 [00:28<00:20, 8440.80it/s] 58%|█████▊    | 230854/400000 [00:28<00:20, 8451.17it/s] 58%|█████▊    | 231700/400000 [00:28<00:19, 8439.66it/s] 58%|█████▊    | 232544/400000 [00:28<00:19, 8412.72it/s] 58%|█████▊    | 233389/400000 [00:28<00:19, 8421.02it/s] 59%|█████▊    | 234232/400000 [00:28<00:19, 8419.80it/s] 59%|█████▉    | 235075/400000 [00:28<00:19, 8422.71it/s] 59%|█████▉    | 235918/400000 [00:28<00:19, 8417.85it/s] 59%|█████▉    | 236760/400000 [00:28<00:19, 8382.12it/s] 59%|█████▉    | 237607/400000 [00:28<00:19, 8406.16it/s] 60%|█████▉    | 238455/400000 [00:29<00:19, 8427.47it/s] 60%|█████▉    | 239298/400000 [00:29<00:19, 8424.13it/s] 60%|██████    | 240141/400000 [00:29<00:19, 8411.10it/s] 60%|██████    | 240983/400000 [00:29<00:18, 8394.54it/s] 60%|██████    | 241829/400000 [00:29<00:18, 8411.75it/s] 61%|██████    | 242677/400000 [00:29<00:18, 8431.33it/s] 61%|██████    | 243525/400000 [00:29<00:18, 8445.52it/s] 61%|██████    | 244370/400000 [00:29<00:18, 8444.19it/s] 61%|██████▏   | 245215/400000 [00:29<00:18, 8341.59it/s] 62%|██████▏   | 246061/400000 [00:29<00:18, 8374.55it/s] 62%|██████▏   | 246905/400000 [00:30<00:18, 8393.43it/s] 62%|██████▏   | 247751/400000 [00:30<00:18, 8411.65it/s] 62%|██████▏   | 248593/400000 [00:30<00:18, 8407.88it/s] 62%|██████▏   | 249434/400000 [00:30<00:17, 8401.87it/s] 63%|██████▎   | 250279/400000 [00:30<00:17, 8416.20it/s] 63%|██████▎   | 251121/400000 [00:30<00:17, 8399.35it/s] 63%|██████▎   | 251966/400000 [00:30<00:17, 8414.06it/s] 63%|██████▎   | 252809/400000 [00:30<00:17, 8416.53it/s] 63%|██████▎   | 253651/400000 [00:30<00:17, 8349.06it/s] 64%|██████▎   | 254501/400000 [00:30<00:17, 8391.17it/s] 64%|██████▍   | 255341/400000 [00:31<00:17, 8303.88it/s] 64%|██████▍   | 256172/400000 [00:31<00:17, 8280.57it/s] 64%|██████▍   | 257001/400000 [00:31<00:17, 8258.54it/s] 64%|██████▍   | 257839/400000 [00:31<00:17, 8293.34it/s] 65%|██████▍   | 258682/400000 [00:31<00:16, 8331.93it/s] 65%|██████▍   | 259522/400000 [00:31<00:16, 8351.30it/s] 65%|██████▌   | 260358/400000 [00:31<00:17, 7955.28it/s] 65%|██████▌   | 261194/400000 [00:31<00:17, 8071.92it/s] 66%|██████▌   | 262022/400000 [00:31<00:16, 8131.12it/s] 66%|██████▌   | 262838/400000 [00:31<00:17, 8008.01it/s] 66%|██████▌   | 263674/400000 [00:32<00:16, 8108.56it/s] 66%|██████▌   | 264514/400000 [00:32<00:16, 8190.59it/s] 66%|██████▋   | 265349/400000 [00:32<00:16, 8237.58it/s] 67%|██████▋   | 266174/400000 [00:32<00:16, 8222.12it/s] 67%|██████▋   | 267012/400000 [00:32<00:16, 8268.20it/s] 67%|██████▋   | 267859/400000 [00:32<00:15, 8325.13it/s] 67%|██████▋   | 268693/400000 [00:32<00:15, 8297.56it/s] 67%|██████▋   | 269524/400000 [00:32<00:15, 8283.59it/s] 68%|██████▊   | 270361/400000 [00:32<00:15, 8309.28it/s] 68%|██████▊   | 271197/400000 [00:32<00:15, 8322.67it/s] 68%|██████▊   | 272038/400000 [00:33<00:15, 8346.29it/s] 68%|██████▊   | 272885/400000 [00:33<00:15, 8382.31it/s] 68%|██████▊   | 273724/400000 [00:33<00:15, 8241.28it/s] 69%|██████▊   | 274553/400000 [00:33<00:15, 8253.24it/s] 69%|██████▉   | 275392/400000 [00:33<00:15, 8291.17it/s] 69%|██████▉   | 276239/400000 [00:33<00:14, 8343.16it/s] 69%|██████▉   | 277074/400000 [00:33<00:14, 8331.85it/s] 69%|██████▉   | 277908/400000 [00:33<00:14, 8322.64it/s] 70%|██████▉   | 278741/400000 [00:33<00:14, 8317.29it/s] 70%|██████▉   | 279573/400000 [00:33<00:14, 8280.58it/s] 70%|███████   | 280418/400000 [00:34<00:14, 8328.45it/s] 70%|███████   | 281266/400000 [00:34<00:14, 8370.90it/s] 71%|███████   | 282107/400000 [00:34<00:14, 8380.71it/s] 71%|███████   | 282946/400000 [00:34<00:14, 8265.99it/s] 71%|███████   | 283777/400000 [00:34<00:14, 8276.27it/s] 71%|███████   | 284618/400000 [00:34<00:13, 8313.88it/s] 71%|███████▏  | 285458/400000 [00:34<00:13, 8336.85it/s] 72%|███████▏  | 286302/400000 [00:34<00:13, 8360.19it/s] 72%|███████▏  | 287146/400000 [00:34<00:13, 8382.95it/s] 72%|███████▏  | 287985/400000 [00:34<00:13, 8360.92it/s] 72%|███████▏  | 288830/400000 [00:35<00:13, 8387.05it/s] 72%|███████▏  | 289678/400000 [00:35<00:13, 8411.84it/s] 73%|███████▎  | 290520/400000 [00:35<00:13, 8197.96it/s] 73%|███████▎  | 291342/400000 [00:35<00:13, 8199.67it/s] 73%|███████▎  | 292181/400000 [00:35<00:13, 8253.95it/s] 73%|███████▎  | 293029/400000 [00:35<00:12, 8320.13it/s] 73%|███████▎  | 293874/400000 [00:35<00:12, 8358.44it/s] 74%|███████▎  | 294717/400000 [00:35<00:12, 8377.93it/s] 74%|███████▍  | 295556/400000 [00:35<00:12, 8378.82it/s] 74%|███████▍  | 296395/400000 [00:35<00:12, 8371.86it/s] 74%|███████▍  | 297237/400000 [00:36<00:12, 8384.11it/s] 75%|███████▍  | 298084/400000 [00:36<00:12, 8408.00it/s] 75%|███████▍  | 298925/400000 [00:36<00:12, 8335.65it/s] 75%|███████▍  | 299767/400000 [00:36<00:11, 8358.46it/s] 75%|███████▌  | 300604/400000 [00:36<00:11, 8344.41it/s] 75%|███████▌  | 301451/400000 [00:36<00:11, 8379.02it/s] 76%|███████▌  | 302290/400000 [00:36<00:12, 8070.00it/s] 76%|███████▌  | 303135/400000 [00:36<00:11, 8179.24it/s] 76%|███████▌  | 303981/400000 [00:36<00:11, 8259.58it/s] 76%|███████▌  | 304812/400000 [00:37<00:11, 8271.88it/s] 76%|███████▋  | 305659/400000 [00:37<00:11, 8329.60it/s] 77%|███████▋  | 306502/400000 [00:37<00:11, 8358.23it/s] 77%|███████▋  | 307350/400000 [00:37<00:11, 8393.41it/s] 77%|███████▋  | 308196/400000 [00:37<00:10, 8411.19it/s] 77%|███████▋  | 309038/400000 [00:37<00:10, 8377.96it/s] 77%|███████▋  | 309883/400000 [00:37<00:10, 8397.48it/s] 78%|███████▊  | 310723/400000 [00:37<00:10, 8232.89it/s] 78%|███████▊  | 311553/400000 [00:37<00:10, 8251.88it/s] 78%|███████▊  | 312383/400000 [00:37<00:10, 8264.81it/s] 78%|███████▊  | 313210/400000 [00:38<00:10, 8264.78it/s] 79%|███████▊  | 314057/400000 [00:38<00:10, 8322.39it/s] 79%|███████▊  | 314903/400000 [00:38<00:10, 8360.87it/s] 79%|███████▉  | 315746/400000 [00:38<00:10, 8380.55it/s] 79%|███████▉  | 316585/400000 [00:38<00:09, 8364.74it/s] 79%|███████▉  | 317422/400000 [00:38<00:09, 8315.48it/s] 80%|███████▉  | 318264/400000 [00:38<00:09, 8346.31it/s] 80%|███████▉  | 319108/400000 [00:38<00:09, 8373.22it/s] 80%|███████▉  | 319952/400000 [00:38<00:09, 8392.22it/s] 80%|████████  | 320792/400000 [00:38<00:09, 8361.41it/s] 80%|████████  | 321629/400000 [00:39<00:09, 8340.58it/s] 81%|████████  | 322479/400000 [00:39<00:09, 8385.86it/s] 81%|████████  | 323329/400000 [00:39<00:09, 8416.94it/s] 81%|████████  | 324171/400000 [00:39<00:09, 8408.00it/s] 81%|████████▏ | 325014/400000 [00:39<00:08, 8414.19it/s] 81%|████████▏ | 325856/400000 [00:39<00:08, 8304.14it/s] 82%|████████▏ | 326703/400000 [00:39<00:08, 8351.21it/s] 82%|████████▏ | 327552/400000 [00:39<00:08, 8389.64it/s] 82%|████████▏ | 328392/400000 [00:39<00:08, 8388.36it/s] 82%|████████▏ | 329236/400000 [00:39<00:08, 8403.13it/s] 83%|████████▎ | 330077/400000 [00:40<00:08, 8312.53it/s] 83%|████████▎ | 330913/400000 [00:40<00:08, 8324.85it/s] 83%|████████▎ | 331753/400000 [00:40<00:08, 8346.98it/s] 83%|████████▎ | 332588/400000 [00:40<00:08, 8313.92it/s] 83%|████████▎ | 333426/400000 [00:40<00:07, 8332.90it/s] 84%|████████▎ | 334260/400000 [00:40<00:07, 8321.30it/s] 84%|████████▍ | 335098/400000 [00:40<00:07, 8336.09it/s] 84%|████████▍ | 335942/400000 [00:40<00:07, 8365.25it/s] 84%|████████▍ | 336785/400000 [00:40<00:07, 8383.82it/s] 84%|████████▍ | 337631/400000 [00:40<00:07, 8405.02it/s] 85%|████████▍ | 338472/400000 [00:41<00:07, 8379.92it/s] 85%|████████▍ | 339311/400000 [00:41<00:07, 8229.48it/s] 85%|████████▌ | 340147/400000 [00:41<00:07, 8265.97it/s] 85%|████████▌ | 340975/400000 [00:41<00:07, 8261.63it/s] 85%|████████▌ | 341820/400000 [00:41<00:06, 8315.25it/s] 86%|████████▌ | 342665/400000 [00:41<00:06, 8355.01it/s] 86%|████████▌ | 343501/400000 [00:41<00:06, 8347.19it/s] 86%|████████▌ | 344336/400000 [00:41<00:06, 8057.37it/s] 86%|████████▋ | 345179/400000 [00:41<00:06, 8164.79it/s] 86%|████████▋ | 345998/400000 [00:41<00:06, 8065.89it/s] 87%|████████▋ | 346807/400000 [00:42<00:06, 7666.82it/s] 87%|████████▋ | 347579/400000 [00:42<00:06, 7569.39it/s] 87%|████████▋ | 348421/400000 [00:42<00:06, 7803.60it/s] 87%|████████▋ | 349268/400000 [00:42<00:06, 7990.02it/s] 88%|████████▊ | 350093/400000 [00:42<00:06, 8064.22it/s] 88%|████████▊ | 350922/400000 [00:42<00:06, 8129.11it/s] 88%|████████▊ | 351743/400000 [00:42<00:05, 8151.83it/s] 88%|████████▊ | 352585/400000 [00:42<00:05, 8229.44it/s] 88%|████████▊ | 353418/400000 [00:42<00:05, 8257.00it/s] 89%|████████▊ | 354253/400000 [00:42<00:05, 8282.22it/s] 89%|████████▉ | 355097/400000 [00:43<00:05, 8326.17it/s] 89%|████████▉ | 355931/400000 [00:43<00:05, 8298.47it/s] 89%|████████▉ | 356779/400000 [00:43<00:05, 8351.95it/s] 89%|████████▉ | 357624/400000 [00:43<00:05, 8380.03it/s] 90%|████████▉ | 358463/400000 [00:43<00:04, 8360.45it/s] 90%|████████▉ | 359307/400000 [00:43<00:04, 8381.15it/s] 90%|█████████ | 360146/400000 [00:43<00:04, 8328.47it/s] 90%|█████████ | 360980/400000 [00:43<00:04, 8211.92it/s] 90%|█████████ | 361821/400000 [00:43<00:04, 8268.96it/s] 91%|█████████ | 362655/400000 [00:43<00:04, 8286.71it/s] 91%|█████████ | 363489/400000 [00:44<00:04, 8301.47it/s] 91%|█████████ | 364320/400000 [00:44<00:04, 8292.93it/s] 91%|█████████▏| 365167/400000 [00:44<00:04, 8344.59it/s] 92%|█████████▏| 366007/400000 [00:44<00:04, 8359.77it/s] 92%|█████████▏| 366844/400000 [00:44<00:04, 8218.28it/s] 92%|█████████▏| 367685/400000 [00:44<00:03, 8274.49it/s] 92%|█████████▏| 368519/400000 [00:44<00:03, 8292.32it/s] 92%|█████████▏| 369366/400000 [00:44<00:03, 8343.65it/s] 93%|█████████▎| 370214/400000 [00:44<00:03, 8382.02it/s] 93%|█████████▎| 371062/400000 [00:44<00:03, 8408.95it/s] 93%|█████████▎| 371904/400000 [00:45<00:03, 8393.07it/s] 93%|█████████▎| 372744/400000 [00:45<00:03, 8377.36it/s] 93%|█████████▎| 373591/400000 [00:45<00:03, 8402.35it/s] 94%|█████████▎| 374432/400000 [00:45<00:03, 8388.55it/s] 94%|█████████▍| 375280/400000 [00:45<00:02, 8413.54it/s] 94%|█████████▍| 376124/400000 [00:45<00:02, 8420.41it/s] 94%|█████████▍| 376967/400000 [00:45<00:02, 8350.61it/s] 94%|█████████▍| 377814/400000 [00:45<00:02, 8384.39it/s] 95%|█████████▍| 378659/400000 [00:45<00:02, 8403.96it/s] 95%|█████████▍| 379505/400000 [00:46<00:02, 8420.13it/s] 95%|█████████▌| 380354/400000 [00:46<00:02, 8438.23it/s] 95%|█████████▌| 381198/400000 [00:46<00:02, 8347.70it/s] 96%|█████████▌| 382046/400000 [00:46<00:02, 8385.79it/s] 96%|█████████▌| 382895/400000 [00:46<00:02, 8414.59it/s] 96%|█████████▌| 383737/400000 [00:46<00:01, 8384.59it/s] 96%|█████████▌| 384576/400000 [00:46<00:01, 8355.51it/s] 96%|█████████▋| 385412/400000 [00:46<00:01, 8311.58it/s] 97%|█████████▋| 386251/400000 [00:46<00:01, 8334.32it/s] 97%|█████████▋| 387085/400000 [00:46<00:01, 8325.10it/s] 97%|█████████▋| 387924/400000 [00:47<00:01, 8342.83it/s] 97%|█████████▋| 388759/400000 [00:47<00:01, 8246.59it/s] 97%|█████████▋| 389604/400000 [00:47<00:01, 8306.09it/s] 98%|█████████▊| 390446/400000 [00:47<00:01, 8339.50it/s] 98%|█████████▊| 391293/400000 [00:47<00:01, 8375.28it/s] 98%|█████████▊| 392136/400000 [00:47<00:00, 8388.98it/s] 98%|█████████▊| 392981/400000 [00:47<00:00, 8406.45it/s] 98%|█████████▊| 393822/400000 [00:47<00:00, 8319.17it/s] 99%|█████████▊| 394656/400000 [00:47<00:00, 8325.02it/s] 99%|█████████▉| 395502/400000 [00:47<00:00, 8364.79it/s] 99%|█████████▉| 396345/400000 [00:48<00:00, 8383.03it/s] 99%|█████████▉| 397184/400000 [00:48<00:00, 8373.19it/s]100%|█████████▉| 398022/400000 [00:48<00:00, 8359.66it/s]100%|█████████▉| 398867/400000 [00:48<00:00, 8385.66it/s]100%|█████████▉| 399710/400000 [00:48<00:00, 8397.50it/s]100%|█████████▉| 399999/400000 [00:48<00:00, 8255.41it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f6910f0b400>, <torchtext.data.dataset.TabularDataset object at 0x7f6910f0b550>, <torchtext.vocab.Vocab object at 0x7f6910f0b470>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  4.93 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  4.93 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  4.93 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  4.93 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.07 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.28 file/s]2020-06-16 06:19:08.809844: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-06-16 06:19:08.814329: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095139999 Hz
2020-06-16 06:19:08.814517: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55c1a7e857b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-16 06:19:08.814531: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:15, 131937.43it/s] 84%|████████▍ | 8339456/9912422 [00:00<00:08, 188353.10it/s]9920512it [00:00, 41567410.67it/s]                           
0it [00:00, ?it/s]32768it [00:00, 686031.94it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 469027.97it/s]1654784it [00:00, 11784535.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 267091.65it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

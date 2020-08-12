
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f031747f510> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f031747f510> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f038212f488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f038212f488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f03a134dea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f03a134dea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f032f468158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f032f468158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f032f468158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 44%|████▍     | 4382720/9912422 [00:00<00:00, 43646901.62it/s]9920512it [00:00, 32447809.47it/s]                             
0it [00:00, ?it/s]32768it [00:00, 554764.23it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 138878.29it/s]1654784it [00:00, 10233892.97it/s]                         
0it [00:00, ?it/s]8192it [00:00, 164359.77it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f03166ba6d8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f032d853978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f032f45fd90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f032f45fd90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f032f45fd90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<00:59,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<00:58,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:57,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:40,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:39,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:39,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:39,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:39,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:38,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.83 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:27,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:26,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:26,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:26,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:25,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.36 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:18,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:18,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:17,  7.47 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:12, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:12, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:12, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:12, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:11, 10.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 14.00 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:08, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:00<00:08, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:00<00:08, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:00<00:08, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:00<00:08, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:00<00:08, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:00<00:08, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:00<00:07, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:00<00:07, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:00<00:07, 14.00 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:00<00:05, 18.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:00<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 18.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.39 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.39 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.64 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.64 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 46.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.09 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.09 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 67.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 67.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:01<00:00, 72.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 76.95 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:01<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:01<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:01<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:01<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:01<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:01<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 76.95 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 80.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 80.81 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 83.60 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.22s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 83.60 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.68 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.54s/ url]
0 examples [00:00, ? examples/s]2020-08-12 18:07:46.351142: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-12 18:07:46.361807: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-12 18:07:46.363172: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55f17e80b010 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-12 18:07:46.363193: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
45 examples [00:00, 449.10 examples/s]136 examples [00:00, 529.14 examples/s]228 examples [00:00, 606.09 examples/s]319 examples [00:00, 672.12 examples/s]412 examples [00:00, 731.77 examples/s]501 examples [00:00, 772.19 examples/s]591 examples [00:00, 806.38 examples/s]680 examples [00:00, 828.29 examples/s]775 examples [00:00, 859.71 examples/s]868 examples [00:01, 879.59 examples/s]962 examples [00:01, 895.95 examples/s]1059 examples [00:01, 916.59 examples/s]1152 examples [00:01, 909.38 examples/s]1247 examples [00:01, 920.14 examples/s]1340 examples [00:01, 922.27 examples/s]1433 examples [00:01, 919.39 examples/s]1526 examples [00:01, 914.11 examples/s]1620 examples [00:01, 919.34 examples/s]1713 examples [00:01, 878.17 examples/s]1803 examples [00:02, 884.43 examples/s]1893 examples [00:02, 888.01 examples/s]1983 examples [00:02, 876.46 examples/s]2071 examples [00:02, 866.57 examples/s]2160 examples [00:02, 872.50 examples/s]2249 examples [00:02, 877.46 examples/s]2337 examples [00:02, 868.92 examples/s]2429 examples [00:02, 883.29 examples/s]2518 examples [00:02, 877.24 examples/s]2606 examples [00:02, 855.34 examples/s]2701 examples [00:03, 880.93 examples/s]2797 examples [00:03, 901.64 examples/s]2890 examples [00:03, 909.82 examples/s]2984 examples [00:03, 915.18 examples/s]3077 examples [00:03, 919.06 examples/s]3170 examples [00:03, 906.48 examples/s]3264 examples [00:03, 914.61 examples/s]3356 examples [00:03, 896.31 examples/s]3446 examples [00:03, 894.25 examples/s]3539 examples [00:03, 904.51 examples/s]3634 examples [00:04, 916.25 examples/s]3729 examples [00:04, 925.23 examples/s]3822 examples [00:04, 906.43 examples/s]3913 examples [00:04, 894.36 examples/s]4003 examples [00:04, 888.35 examples/s]4096 examples [00:04, 898.57 examples/s]4190 examples [00:04, 909.66 examples/s]4282 examples [00:04, 906.07 examples/s]4374 examples [00:04, 909.48 examples/s]4466 examples [00:05, 893.65 examples/s]4559 examples [00:05, 903.94 examples/s]4655 examples [00:05, 919.03 examples/s]4749 examples [00:05, 922.27 examples/s]4842 examples [00:05, 923.29 examples/s]4935 examples [00:05, 910.07 examples/s]5027 examples [00:05, 903.59 examples/s]5118 examples [00:05, 903.88 examples/s]5210 examples [00:05, 907.51 examples/s]5304 examples [00:05, 915.88 examples/s]5396 examples [00:06, 909.22 examples/s]5487 examples [00:06, 890.55 examples/s]5581 examples [00:06, 904.52 examples/s]5675 examples [00:06, 912.12 examples/s]5770 examples [00:06, 922.40 examples/s]5865 examples [00:06, 928.79 examples/s]5958 examples [00:06, 922.30 examples/s]6051 examples [00:06, 920.27 examples/s]6148 examples [00:06, 931.96 examples/s]6242 examples [00:06, 929.68 examples/s]6336 examples [00:07, 931.54 examples/s]6430 examples [00:07, 928.11 examples/s]6524 examples [00:07, 929.87 examples/s]6618 examples [00:07, 923.46 examples/s]6711 examples [00:07, 902.04 examples/s]6806 examples [00:07, 915.85 examples/s]6898 examples [00:07, 892.81 examples/s]6994 examples [00:07, 910.37 examples/s]7088 examples [00:07, 916.76 examples/s]7181 examples [00:07, 920.32 examples/s]7276 examples [00:08, 927.20 examples/s]7369 examples [00:08, 920.68 examples/s]7462 examples [00:08, 917.42 examples/s]7555 examples [00:08, 918.57 examples/s]7648 examples [00:08, 920.21 examples/s]7741 examples [00:08, 922.79 examples/s]7835 examples [00:08, 926.23 examples/s]7928 examples [00:08, 925.86 examples/s]8021 examples [00:08, 925.10 examples/s]8115 examples [00:08, 928.23 examples/s]8213 examples [00:09, 942.07 examples/s]8308 examples [00:09, 933.18 examples/s]8403 examples [00:09, 936.55 examples/s]8500 examples [00:09, 944.12 examples/s]8598 examples [00:09, 951.47 examples/s]8694 examples [00:09, 953.23 examples/s]8790 examples [00:09, 947.11 examples/s]8886 examples [00:09, 950.75 examples/s]8985 examples [00:09, 959.82 examples/s]9082 examples [00:09, 955.03 examples/s]9181 examples [00:10, 964.77 examples/s]9278 examples [00:10, 953.82 examples/s]9374 examples [00:10, 955.50 examples/s]9470 examples [00:10, 950.04 examples/s]9566 examples [00:10, 951.86 examples/s]9665 examples [00:10, 959.37 examples/s]9761 examples [00:10, 948.89 examples/s]9856 examples [00:10, 933.72 examples/s]9953 examples [00:10, 943.11 examples/s]10048 examples [00:11, 866.39 examples/s]10141 examples [00:11, 883.13 examples/s]10236 examples [00:11, 901.46 examples/s]10334 examples [00:11, 921.48 examples/s]10429 examples [00:11, 928.83 examples/s]10524 examples [00:11, 932.83 examples/s]10620 examples [00:11, 940.15 examples/s]10715 examples [00:11, 941.93 examples/s]10816 examples [00:11, 958.86 examples/s]10913 examples [00:11, 951.33 examples/s]11009 examples [00:12, 946.89 examples/s]11106 examples [00:12, 951.76 examples/s]11202 examples [00:12, 944.44 examples/s]11297 examples [00:12, 933.79 examples/s]11391 examples [00:12, 919.42 examples/s]11485 examples [00:12, 922.60 examples/s]11578 examples [00:12, 924.53 examples/s]11671 examples [00:12, 913.66 examples/s]11766 examples [00:12, 921.74 examples/s]11860 examples [00:12, 925.27 examples/s]11953 examples [00:13, 922.67 examples/s]12049 examples [00:13, 931.67 examples/s]12143 examples [00:13, 931.98 examples/s]12237 examples [00:13, 927.61 examples/s]12330 examples [00:13, 921.41 examples/s]12425 examples [00:13, 928.17 examples/s]12519 examples [00:13, 930.01 examples/s]12613 examples [00:13, 929.94 examples/s]12707 examples [00:13, 928.76 examples/s]12801 examples [00:13, 930.75 examples/s]12895 examples [00:14, 915.75 examples/s]12987 examples [00:14, 917.00 examples/s]13079 examples [00:14, 894.10 examples/s]13172 examples [00:14, 903.54 examples/s]13263 examples [00:14, 902.72 examples/s]13354 examples [00:14, 883.69 examples/s]13444 examples [00:14, 886.01 examples/s]13535 examples [00:14, 890.16 examples/s]13625 examples [00:14, 885.29 examples/s]13715 examples [00:15, 889.42 examples/s]13807 examples [00:15, 896.14 examples/s]13900 examples [00:15, 903.80 examples/s]13991 examples [00:15, 900.23 examples/s]14085 examples [00:15, 910.43 examples/s]14177 examples [00:15, 904.45 examples/s]14268 examples [00:15, 869.04 examples/s]14356 examples [00:15, 849.79 examples/s]14446 examples [00:15, 863.26 examples/s]14540 examples [00:15, 883.52 examples/s]14636 examples [00:16, 904.09 examples/s]14727 examples [00:16, 905.73 examples/s]14818 examples [00:16, 905.92 examples/s]14913 examples [00:16, 917.83 examples/s]15011 examples [00:16, 933.92 examples/s]15105 examples [00:16, 924.03 examples/s]15200 examples [00:16, 931.06 examples/s]15294 examples [00:16, 923.62 examples/s]15392 examples [00:16, 937.91 examples/s]15486 examples [00:16, 928.79 examples/s]15579 examples [00:17, 926.21 examples/s]15674 examples [00:17, 931.89 examples/s]15768 examples [00:17, 925.41 examples/s]15861 examples [00:17, 920.41 examples/s]15958 examples [00:17, 934.12 examples/s]16053 examples [00:17, 936.34 examples/s]16149 examples [00:17, 941.78 examples/s]16244 examples [00:17, 931.97 examples/s]16339 examples [00:17, 934.79 examples/s]16433 examples [00:17, 932.26 examples/s]16527 examples [00:18, 934.26 examples/s]16621 examples [00:18, 919.21 examples/s]16713 examples [00:18, 911.37 examples/s]16805 examples [00:18, 912.39 examples/s]16900 examples [00:18, 922.93 examples/s]16995 examples [00:18, 929.93 examples/s]17089 examples [00:18, 922.20 examples/s]17182 examples [00:18, 911.20 examples/s]17274 examples [00:18, 886.62 examples/s]17368 examples [00:18, 901.44 examples/s]17465 examples [00:19, 920.19 examples/s]17559 examples [00:19, 924.25 examples/s]17652 examples [00:19, 919.48 examples/s]17747 examples [00:19, 928.13 examples/s]17842 examples [00:19, 932.53 examples/s]17936 examples [00:19, 934.61 examples/s]18030 examples [00:19, 931.41 examples/s]18124 examples [00:19, 923.48 examples/s]18217 examples [00:19, 910.30 examples/s]18309 examples [00:20, 872.06 examples/s]18401 examples [00:20, 884.15 examples/s]18490 examples [00:20, 871.14 examples/s]18582 examples [00:20, 883.98 examples/s]18675 examples [00:20, 895.46 examples/s]18768 examples [00:20, 905.41 examples/s]18863 examples [00:20, 917.10 examples/s]18955 examples [00:20, 908.74 examples/s]19046 examples [00:20, 907.63 examples/s]19143 examples [00:20, 923.39 examples/s]19236 examples [00:21, 917.38 examples/s]19328 examples [00:21, 916.92 examples/s]19420 examples [00:21, 915.35 examples/s]19512 examples [00:21, 907.59 examples/s]19603 examples [00:21, 877.43 examples/s]19691 examples [00:21, 870.48 examples/s]19783 examples [00:21, 883.35 examples/s]19873 examples [00:21, 886.48 examples/s]19964 examples [00:21, 892.10 examples/s]20054 examples [00:21, 843.09 examples/s]20142 examples [00:22, 853.47 examples/s]20237 examples [00:22, 879.99 examples/s]20330 examples [00:22, 892.97 examples/s]20420 examples [00:22, 892.22 examples/s]20510 examples [00:22, 877.39 examples/s]20599 examples [00:22, 867.22 examples/s]20691 examples [00:22, 880.77 examples/s]20785 examples [00:22, 894.56 examples/s]20875 examples [00:22, 895.03 examples/s]20969 examples [00:23, 907.99 examples/s]21062 examples [00:23, 914.13 examples/s]21160 examples [00:23, 932.42 examples/s]21256 examples [00:23, 937.49 examples/s]21350 examples [00:23, 932.95 examples/s]21444 examples [00:23, 931.44 examples/s]21538 examples [00:23, 918.27 examples/s]21632 examples [00:23, 922.22 examples/s]21725 examples [00:23, 919.97 examples/s]21818 examples [00:23, 902.94 examples/s]21911 examples [00:24, 909.62 examples/s]22007 examples [00:24, 923.28 examples/s]22100 examples [00:24, 921.66 examples/s]22193 examples [00:24, 910.37 examples/s]22285 examples [00:24, 911.35 examples/s]22380 examples [00:24, 920.68 examples/s]22475 examples [00:24, 927.78 examples/s]22568 examples [00:24, 921.20 examples/s]22661 examples [00:24, 911.22 examples/s]22753 examples [00:24, 900.35 examples/s]22846 examples [00:25, 906.80 examples/s]22940 examples [00:25, 915.19 examples/s]23032 examples [00:25, 910.37 examples/s]23124 examples [00:25, 903.25 examples/s]23217 examples [00:25, 908.27 examples/s]23308 examples [00:25, 902.66 examples/s]23400 examples [00:25, 905.06 examples/s]23491 examples [00:25, 886.36 examples/s]23580 examples [00:25, 881.47 examples/s]23672 examples [00:25, 889.60 examples/s]23764 examples [00:26, 896.16 examples/s]23857 examples [00:26, 905.72 examples/s]23948 examples [00:26, 890.98 examples/s]24041 examples [00:26, 901.21 examples/s]24132 examples [00:26, 900.78 examples/s]24224 examples [00:26, 905.88 examples/s]24318 examples [00:26, 912.75 examples/s]24410 examples [00:26, 886.98 examples/s]24499 examples [00:26, 887.40 examples/s]24588 examples [00:26, 880.02 examples/s]24685 examples [00:27, 904.63 examples/s]24783 examples [00:27, 925.00 examples/s]24877 examples [00:27, 929.31 examples/s]24971 examples [00:27, 920.80 examples/s]25064 examples [00:27, 916.73 examples/s]25159 examples [00:27, 926.33 examples/s]25253 examples [00:27, 928.64 examples/s]25346 examples [00:27, 920.12 examples/s]25441 examples [00:27, 926.63 examples/s]25534 examples [00:28, 896.60 examples/s]25624 examples [00:28, 893.38 examples/s]25714 examples [00:28, 888.58 examples/s]25803 examples [00:28, 884.69 examples/s]25895 examples [00:28, 892.23 examples/s]25986 examples [00:28, 895.57 examples/s]26083 examples [00:28, 914.79 examples/s]26175 examples [00:28, 903.16 examples/s]26266 examples [00:28, 889.82 examples/s]26359 examples [00:28, 899.98 examples/s]26450 examples [00:29, 902.96 examples/s]26541 examples [00:29, 899.80 examples/s]26636 examples [00:29, 912.33 examples/s]26728 examples [00:29, 911.99 examples/s]26828 examples [00:29, 934.35 examples/s]26926 examples [00:29, 947.21 examples/s]27021 examples [00:29, 935.30 examples/s]27115 examples [00:29, 924.29 examples/s]27208 examples [00:29, 879.18 examples/s]27298 examples [00:29, 884.94 examples/s]27389 examples [00:30, 890.98 examples/s]27479 examples [00:30, 891.90 examples/s]27572 examples [00:30, 901.98 examples/s]27668 examples [00:30, 918.05 examples/s]27762 examples [00:30, 919.81 examples/s]27857 examples [00:30, 927.74 examples/s]27958 examples [00:30, 949.35 examples/s]28056 examples [00:30, 955.09 examples/s]28152 examples [00:30, 928.50 examples/s]28247 examples [00:30, 934.72 examples/s]28343 examples [00:31, 939.63 examples/s]28439 examples [00:31, 943.97 examples/s]28534 examples [00:31, 941.71 examples/s]28629 examples [00:31, 919.97 examples/s]28722 examples [00:31, 915.67 examples/s]28815 examples [00:31, 917.15 examples/s]28907 examples [00:31, 895.67 examples/s]28997 examples [00:31, 895.70 examples/s]29092 examples [00:31, 911.06 examples/s]29184 examples [00:32, 910.46 examples/s]29276 examples [00:32, 904.65 examples/s]29370 examples [00:32, 912.14 examples/s]29462 examples [00:32, 906.84 examples/s]29553 examples [00:32, 889.66 examples/s]29643 examples [00:32, 881.01 examples/s]29736 examples [00:32, 893.36 examples/s]29830 examples [00:32, 906.02 examples/s]29927 examples [00:32, 923.34 examples/s]30020 examples [00:32, 872.47 examples/s]30109 examples [00:33, 877.62 examples/s]30202 examples [00:33, 890.58 examples/s]30294 examples [00:33, 896.31 examples/s]30388 examples [00:33, 907.19 examples/s]30485 examples [00:33, 924.60 examples/s]30578 examples [00:33, 921.62 examples/s]30671 examples [00:33, 902.79 examples/s]30764 examples [00:33, 910.09 examples/s]30856 examples [00:33, 910.07 examples/s]30948 examples [00:33, 911.11 examples/s]31040 examples [00:34, 908.57 examples/s]31132 examples [00:34, 910.98 examples/s]31225 examples [00:34, 915.59 examples/s]31317 examples [00:34, 910.62 examples/s]31418 examples [00:34, 934.90 examples/s]31512 examples [00:34, 919.85 examples/s]31605 examples [00:34, 918.64 examples/s]31699 examples [00:34, 924.57 examples/s]31792 examples [00:34, 901.05 examples/s]31883 examples [00:34, 898.97 examples/s]31974 examples [00:35, 891.36 examples/s]32070 examples [00:35, 909.43 examples/s]32162 examples [00:35, 899.27 examples/s]32258 examples [00:35, 914.08 examples/s]32350 examples [00:35, 896.40 examples/s]32440 examples [00:35, 893.68 examples/s]32530 examples [00:35, 890.06 examples/s]32620 examples [00:35, 876.44 examples/s]32717 examples [00:35, 900.61 examples/s]32810 examples [00:36, 907.74 examples/s]32906 examples [00:36, 921.79 examples/s]33005 examples [00:36, 939.41 examples/s]33100 examples [00:36, 941.02 examples/s]33195 examples [00:36, 943.27 examples/s]33290 examples [00:36, 935.93 examples/s]33384 examples [00:36, 918.76 examples/s]33481 examples [00:36, 931.82 examples/s]33578 examples [00:36, 942.06 examples/s]33673 examples [00:36, 942.18 examples/s]33769 examples [00:37, 947.42 examples/s]33864 examples [00:37, 925.36 examples/s]33961 examples [00:37, 936.89 examples/s]34056 examples [00:37, 938.48 examples/s]34153 examples [00:37, 946.15 examples/s]34248 examples [00:37, 939.68 examples/s]34343 examples [00:37, 941.65 examples/s]34439 examples [00:37, 944.48 examples/s]34534 examples [00:37, 943.92 examples/s]34632 examples [00:37, 952.43 examples/s]34728 examples [00:38, 953.59 examples/s]34824 examples [00:38, 942.80 examples/s]34919 examples [00:38, 935.61 examples/s]35017 examples [00:38, 945.96 examples/s]35115 examples [00:38, 954.47 examples/s]35212 examples [00:38, 956.41 examples/s]35308 examples [00:38, 944.52 examples/s]35403 examples [00:38, 941.26 examples/s]35498 examples [00:38, 930.77 examples/s]35592 examples [00:38, 931.15 examples/s]35686 examples [00:39, 926.71 examples/s]35780 examples [00:39, 928.36 examples/s]35873 examples [00:39, 927.63 examples/s]35966 examples [00:39, 927.13 examples/s]36061 examples [00:39, 931.32 examples/s]36155 examples [00:39, 913.03 examples/s]36247 examples [00:39, 907.68 examples/s]36340 examples [00:39, 914.15 examples/s]36439 examples [00:39, 934.32 examples/s]36535 examples [00:39, 941.43 examples/s]36631 examples [00:40, 944.69 examples/s]36726 examples [00:40, 927.68 examples/s]36819 examples [00:40, 927.76 examples/s]36912 examples [00:40, 926.50 examples/s]37007 examples [00:40, 933.11 examples/s]37103 examples [00:40, 938.07 examples/s]37197 examples [00:40, 928.97 examples/s]37297 examples [00:40, 946.96 examples/s]37392 examples [00:40, 922.20 examples/s]37486 examples [00:41, 925.81 examples/s]37581 examples [00:41, 932.83 examples/s]37675 examples [00:41, 929.44 examples/s]37773 examples [00:41, 942.33 examples/s]37873 examples [00:41, 955.94 examples/s]37972 examples [00:41, 965.38 examples/s]38069 examples [00:41, 955.58 examples/s]38165 examples [00:41, 927.39 examples/s]38263 examples [00:41, 940.54 examples/s]38359 examples [00:41, 946.26 examples/s]38455 examples [00:42, 948.42 examples/s]38550 examples [00:42, 935.63 examples/s]38644 examples [00:42, 928.56 examples/s]38738 examples [00:42, 931.12 examples/s]38832 examples [00:42, 919.46 examples/s]38925 examples [00:42, 914.01 examples/s]39022 examples [00:42, 927.20 examples/s]39115 examples [00:42, 926.75 examples/s]39210 examples [00:42, 932.14 examples/s]39307 examples [00:42, 941.48 examples/s]39402 examples [00:43, 935.03 examples/s]39501 examples [00:43, 950.30 examples/s]39597 examples [00:43, 939.42 examples/s]39692 examples [00:43, 931.16 examples/s]39786 examples [00:43, 924.69 examples/s]39879 examples [00:43, 924.46 examples/s]39972 examples [00:43, 925.47 examples/s]40065 examples [00:43, 856.27 examples/s]40156 examples [00:43, 869.92 examples/s]40251 examples [00:43, 891.07 examples/s]40341 examples [00:44, 890.97 examples/s]40431 examples [00:44, 891.50 examples/s]40521 examples [00:44, 883.72 examples/s]40617 examples [00:44, 903.73 examples/s]40716 examples [00:44, 927.46 examples/s]40810 examples [00:44, 930.28 examples/s]40904 examples [00:44, 927.21 examples/s]40997 examples [00:44, 921.53 examples/s]41094 examples [00:44, 932.79 examples/s]41188 examples [00:45, 929.80 examples/s]41282 examples [00:45, 899.21 examples/s]41375 examples [00:45, 906.83 examples/s]41466 examples [00:45, 905.65 examples/s]41563 examples [00:45, 921.99 examples/s]41657 examples [00:45, 925.28 examples/s]41750 examples [00:45, 909.15 examples/s]41842 examples [00:45, 894.71 examples/s]41933 examples [00:45, 897.02 examples/s]42023 examples [00:45, 894.86 examples/s]42113 examples [00:46, 895.40 examples/s]42203 examples [00:46, 890.80 examples/s]42298 examples [00:46, 907.10 examples/s]42394 examples [00:46, 921.19 examples/s]42487 examples [00:46, 895.12 examples/s]42580 examples [00:46, 905.27 examples/s]42678 examples [00:46, 924.00 examples/s]42771 examples [00:46, 923.40 examples/s]42866 examples [00:46, 930.38 examples/s]42963 examples [00:46, 940.61 examples/s]43058 examples [00:47, 937.96 examples/s]43159 examples [00:47, 957.39 examples/s]43255 examples [00:47, 942.22 examples/s]43350 examples [00:47, 927.75 examples/s]43443 examples [00:47, 922.77 examples/s]43536 examples [00:47, 909.94 examples/s]43632 examples [00:47, 922.72 examples/s]43725 examples [00:47, 923.97 examples/s]43821 examples [00:47, 934.44 examples/s]43918 examples [00:47, 943.05 examples/s]44013 examples [00:48, 938.20 examples/s]44107 examples [00:48, 915.23 examples/s]44199 examples [00:48, 896.29 examples/s]44296 examples [00:48, 914.64 examples/s]44389 examples [00:48, 918.02 examples/s]44481 examples [00:48, 910.79 examples/s]44575 examples [00:48, 919.25 examples/s]44668 examples [00:48, 909.53 examples/s]44760 examples [00:48, 911.43 examples/s]44855 examples [00:49, 920.48 examples/s]44948 examples [00:49, 916.53 examples/s]45040 examples [00:49, 907.27 examples/s]45131 examples [00:49, 889.34 examples/s]45225 examples [00:49, 902.28 examples/s]45318 examples [00:49, 909.02 examples/s]45410 examples [00:49, 909.34 examples/s]45502 examples [00:49, 909.73 examples/s]45594 examples [00:49, 889.91 examples/s]45692 examples [00:49, 914.53 examples/s]45790 examples [00:50, 930.03 examples/s]45891 examples [00:50, 952.37 examples/s]45989 examples [00:50, 958.25 examples/s]46086 examples [00:50, 951.04 examples/s]46185 examples [00:50, 960.53 examples/s]46282 examples [00:50, 952.71 examples/s]46380 examples [00:50, 960.63 examples/s]46477 examples [00:50, 927.22 examples/s]46571 examples [00:50, 880.37 examples/s]46662 examples [00:50, 888.58 examples/s]46756 examples [00:51, 901.28 examples/s]46849 examples [00:51, 908.42 examples/s]46942 examples [00:51, 914.72 examples/s]47034 examples [00:51, 905.20 examples/s]47128 examples [00:51, 913.96 examples/s]47221 examples [00:51, 916.79 examples/s]47313 examples [00:51, 910.22 examples/s]47407 examples [00:51, 916.71 examples/s]47499 examples [00:51, 917.41 examples/s]47593 examples [00:51, 922.29 examples/s]47687 examples [00:52, 924.90 examples/s]47780 examples [00:52, 915.06 examples/s]47874 examples [00:52, 921.36 examples/s]47967 examples [00:52, 892.58 examples/s]48058 examples [00:52, 896.54 examples/s]48151 examples [00:52, 903.00 examples/s]48244 examples [00:52, 908.23 examples/s]48335 examples [00:52, 906.42 examples/s]48426 examples [00:52, 882.57 examples/s]48521 examples [00:53, 900.22 examples/s]48613 examples [00:53, 903.02 examples/s]48711 examples [00:53, 923.12 examples/s]48806 examples [00:53, 930.92 examples/s]48900 examples [00:53, 927.91 examples/s]48994 examples [00:53, 931.41 examples/s]49088 examples [00:53, 933.44 examples/s]49182 examples [00:53, 895.58 examples/s]49277 examples [00:53, 907.96 examples/s]49369 examples [00:53, 903.98 examples/s]49464 examples [00:54, 915.18 examples/s]49558 examples [00:54, 919.70 examples/s]49651 examples [00:54, 907.62 examples/s]49742 examples [00:54, 902.27 examples/s]49834 examples [00:54, 905.49 examples/s]49931 examples [00:54, 923.80 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6609/50000 [00:00<00:00, 66088.33 examples/s] 37%|███▋      | 18268/50000 [00:00<00:00, 75958.93 examples/s] 59%|█████▉    | 29398/50000 [00:00<00:00, 83955.97 examples/s] 81%|████████  | 40358/50000 [00:00<00:00, 90293.41 examples/s]                                                               0 examples [00:00, ? examples/s]74 examples [00:00, 738.36 examples/s]170 examples [00:00, 792.88 examples/s]264 examples [00:00, 830.90 examples/s]353 examples [00:00, 845.74 examples/s]441 examples [00:00, 855.03 examples/s]531 examples [00:00, 866.20 examples/s]625 examples [00:00, 885.36 examples/s]716 examples [00:00, 891.11 examples/s]813 examples [00:00, 911.27 examples/s]904 examples [00:01, 894.71 examples/s]999 examples [00:01, 908.37 examples/s]1093 examples [00:01, 917.31 examples/s]1184 examples [00:01, 908.19 examples/s]1275 examples [00:01, 895.83 examples/s]1368 examples [00:01, 905.38 examples/s]1459 examples [00:01, 876.19 examples/s]1554 examples [00:01, 894.85 examples/s]1648 examples [00:01, 905.24 examples/s]1739 examples [00:01, 901.57 examples/s]1835 examples [00:02, 915.92 examples/s]1928 examples [00:02, 917.59 examples/s]2021 examples [00:02, 921.00 examples/s]2114 examples [00:02, 922.22 examples/s]2211 examples [00:02, 934.40 examples/s]2307 examples [00:02, 939.37 examples/s]2401 examples [00:02, 922.00 examples/s]2494 examples [00:02, 905.29 examples/s]2585 examples [00:02, 894.52 examples/s]2678 examples [00:02, 902.70 examples/s]2771 examples [00:03, 908.65 examples/s]2864 examples [00:03, 913.04 examples/s]2959 examples [00:03, 923.04 examples/s]3052 examples [00:03, 922.83 examples/s]3150 examples [00:03, 936.88 examples/s]3244 examples [00:03, 934.78 examples/s]3339 examples [00:03, 936.66 examples/s]3434 examples [00:03, 940.32 examples/s]3529 examples [00:03, 932.29 examples/s]3627 examples [00:03, 945.06 examples/s]3726 examples [00:04, 956.92 examples/s]3827 examples [00:04, 971.28 examples/s]3925 examples [00:04, 967.45 examples/s]4022 examples [00:04, 960.37 examples/s]4119 examples [00:04, 956.20 examples/s]4216 examples [00:04, 957.90 examples/s]4312 examples [00:04, 954.35 examples/s]4408 examples [00:04, 946.79 examples/s]4503 examples [00:04, 929.26 examples/s]4598 examples [00:04, 933.73 examples/s]4692 examples [00:05, 928.05 examples/s]4785 examples [00:05, 909.13 examples/s]4880 examples [00:05, 920.97 examples/s]4973 examples [00:05, 919.77 examples/s]5066 examples [00:05, 922.10 examples/s]5159 examples [00:05, 921.80 examples/s]5252 examples [00:05, 922.69 examples/s]5347 examples [00:05, 928.47 examples/s]5440 examples [00:05, 923.38 examples/s]5533 examples [00:06, 922.31 examples/s]5628 examples [00:06, 928.54 examples/s]5721 examples [00:06, 926.75 examples/s]5817 examples [00:06, 935.49 examples/s]5911 examples [00:06, 930.49 examples/s]6006 examples [00:06, 936.03 examples/s]6101 examples [00:06, 938.56 examples/s]6197 examples [00:06, 942.32 examples/s]6293 examples [00:06, 946.95 examples/s]6388 examples [00:06, 937.76 examples/s]6483 examples [00:07, 940.98 examples/s]6580 examples [00:07, 948.32 examples/s]6675 examples [00:07, 946.57 examples/s]6771 examples [00:07, 948.54 examples/s]6866 examples [00:07, 926.38 examples/s]6962 examples [00:07, 933.68 examples/s]7056 examples [00:07, 919.07 examples/s]7150 examples [00:07, 923.52 examples/s]7247 examples [00:07, 935.74 examples/s]7341 examples [00:07, 935.74 examples/s]7435 examples [00:08, 935.07 examples/s]7532 examples [00:08, 943.39 examples/s]7627 examples [00:08, 925.15 examples/s]7720 examples [00:08, 900.93 examples/s]7817 examples [00:08, 920.19 examples/s]7913 examples [00:08, 930.70 examples/s]8012 examples [00:08, 945.68 examples/s]8107 examples [00:08, 943.42 examples/s]8202 examples [00:08, 942.26 examples/s]8297 examples [00:08, 918.27 examples/s]8390 examples [00:09, 899.94 examples/s]8481 examples [00:09, 900.79 examples/s]8572 examples [00:09, 880.46 examples/s]8661 examples [00:09, 882.30 examples/s]8750 examples [00:09, 877.85 examples/s]8842 examples [00:09, 887.40 examples/s]8938 examples [00:09, 905.29 examples/s]9032 examples [00:09, 914.47 examples/s]9124 examples [00:09, 914.28 examples/s]9217 examples [00:09, 918.27 examples/s]9310 examples [00:10, 918.71 examples/s]9407 examples [00:10, 931.77 examples/s]9507 examples [00:10, 949.95 examples/s]9603 examples [00:10, 949.89 examples/s]9699 examples [00:10, 948.63 examples/s]9797 examples [00:10, 956.36 examples/s]9893 examples [00:10, 932.55 examples/s]9987 examples [00:10, 921.24 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCHNXG1/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteCHNXG1/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f032f468158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f032f468158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f032f468158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f037ae480b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f03191b7da0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f032f468158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f032f468158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f032f468158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f032d853358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f032d8531d0>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.64 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.64 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.64 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.52 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.12 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.02 file/s]2020-08-12 18:09:00.210021: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-12 18:09:00.213696: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-08-12 18:09:00.213903: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559299d40c60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-12 18:09:00.213918: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:09, 143131.58it/s] 51%|█████     | 5021696/9912422 [00:00<00:23, 204223.15it/s]9920512it [00:00, 38326481.76it/s]                           
0it [00:00, ?it/s]32768it [00:00, 369319.57it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 137871.59it/s]1654784it [00:00, 9196456.86it/s]                          
0it [00:00, ?it/s]  0%|          | 0/4542 [00:00<?, ?it/s]8192it [00:00, 78584.32it/s]            Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

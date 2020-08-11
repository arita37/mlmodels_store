
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f480fc07598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f480fc07598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f487a8b7488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f487a8b7488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f4899ad5ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f4899ad5ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4827bf2158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4827bf2158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4827bf2158> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:10, 141323.37it/s] 80%|████████  | 7979008/9912422 [00:00<00:09, 201736.77it/s]9920512it [00:00, 41982023.71it/s]                           
0it [00:00, ?it/s]32768it [00:00, 586731.64it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 156112.51it/s]1654784it [00:00, 11250441.14it/s]                         
0it [00:00, ?it/s]8192it [00:00, 215963.16it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4827bec4a8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4825fda9e8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f4827be7d90> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f4827be7d90> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f4827be7d90> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:10,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:10,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:09,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:09,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:09,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:08,  2.28 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.20 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:48,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:47,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:47,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:46,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:46,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:45,  3.20 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:32,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:31,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:31,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:31,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:30,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:30,  4.49 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.27 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:21,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:21,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:21,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:21,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:20,  6.27 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:13,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:13,  8.67 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:10, 11.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:00<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:09, 11.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 15.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 15.94 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 21.03 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 27.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 27.11 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 33.88 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.06 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 41.06 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 54.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 60.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.53 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 65.53 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 70.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 73.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.28 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 74.28 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 76.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 76.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 76.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.45s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.45s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.02 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.45s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 76.02 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.45s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 76.02 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.09s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.84 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.09s/ url]
0 examples [00:00, ? examples/s]2020-08-11 12:08:27.152524: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-11 12:08:27.170117: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-11 12:08:27.170334: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55e8bd7789b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-11 12:08:27.170355: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
5 examples [00:00, 49.50 examples/s]95 examples [00:00, 69.08 examples/s]179 examples [00:00, 95.30 examples/s]265 examples [00:00, 129.95 examples/s]353 examples [00:00, 174.55 examples/s]443 examples [00:00, 230.05 examples/s]530 examples [00:00, 295.08 examples/s]615 examples [00:00, 366.86 examples/s]704 examples [00:00, 444.89 examples/s]796 examples [00:01, 525.82 examples/s]881 examples [00:01, 591.59 examples/s]970 examples [00:01, 657.54 examples/s]1060 examples [00:01, 713.84 examples/s]1147 examples [00:01, 747.74 examples/s]1238 examples [00:01, 788.01 examples/s]1327 examples [00:01, 816.06 examples/s]1418 examples [00:01, 841.41 examples/s]1507 examples [00:01, 843.70 examples/s]1595 examples [00:01, 837.44 examples/s]1682 examples [00:02, 846.17 examples/s]1773 examples [00:02, 862.37 examples/s]1861 examples [00:02, 867.54 examples/s]1949 examples [00:02, 862.58 examples/s]2041 examples [00:02, 877.56 examples/s]2132 examples [00:02, 884.97 examples/s]2221 examples [00:02, 882.70 examples/s]2311 examples [00:02, 887.19 examples/s]2400 examples [00:02, 837.66 examples/s]2485 examples [00:02, 832.28 examples/s]2571 examples [00:03, 838.57 examples/s]2656 examples [00:03, 827.68 examples/s]2740 examples [00:03, 808.02 examples/s]2822 examples [00:03, 804.20 examples/s]2909 examples [00:03, 820.22 examples/s]2997 examples [00:03, 834.62 examples/s]3086 examples [00:03, 850.28 examples/s]3172 examples [00:03, 850.00 examples/s]3262 examples [00:03, 862.32 examples/s]3349 examples [00:03, 853.97 examples/s]3439 examples [00:04, 864.74 examples/s]3531 examples [00:04, 878.54 examples/s]3624 examples [00:04, 890.82 examples/s]3714 examples [00:04, 861.66 examples/s]3805 examples [00:04, 873.14 examples/s]3896 examples [00:04, 883.78 examples/s]3986 examples [00:04, 886.68 examples/s]4076 examples [00:04, 889.53 examples/s]4166 examples [00:04, 878.09 examples/s]4254 examples [00:05, 877.39 examples/s]4344 examples [00:05, 883.27 examples/s]4435 examples [00:05, 889.00 examples/s]4524 examples [00:05, 887.57 examples/s]4614 examples [00:05, 889.94 examples/s]4704 examples [00:05, 886.02 examples/s]4793 examples [00:05, 834.78 examples/s]4882 examples [00:05, 850.59 examples/s]4973 examples [00:05, 867.43 examples/s]5061 examples [00:05, 869.93 examples/s]5149 examples [00:06, 866.21 examples/s]5236 examples [00:06, 856.35 examples/s]5324 examples [00:06, 862.88 examples/s]5412 examples [00:06, 864.61 examples/s]5499 examples [00:06, 858.94 examples/s]5589 examples [00:06, 868.06 examples/s]5677 examples [00:06, 870.91 examples/s]5767 examples [00:06, 876.61 examples/s]5857 examples [00:06, 881.78 examples/s]5946 examples [00:06, 880.67 examples/s]6035 examples [00:07, 876.87 examples/s]6123 examples [00:07, 858.69 examples/s]6213 examples [00:07, 867.87 examples/s]6302 examples [00:07, 871.47 examples/s]6391 examples [00:07, 876.39 examples/s]6479 examples [00:07, 867.21 examples/s]6568 examples [00:07, 871.39 examples/s]6658 examples [00:07, 879.56 examples/s]6749 examples [00:07, 887.84 examples/s]6838 examples [00:07, 872.45 examples/s]6927 examples [00:08, 875.13 examples/s]7015 examples [00:08, 854.42 examples/s]7104 examples [00:08, 864.34 examples/s]7199 examples [00:08, 885.21 examples/s]7290 examples [00:08, 890.46 examples/s]7380 examples [00:08, 880.14 examples/s]7470 examples [00:08, 884.06 examples/s]7559 examples [00:08, 868.62 examples/s]7648 examples [00:08, 872.24 examples/s]7736 examples [00:09, 872.08 examples/s]7824 examples [00:09, 870.17 examples/s]7914 examples [00:09, 878.61 examples/s]8002 examples [00:09, 877.29 examples/s]8090 examples [00:09, 872.63 examples/s]8178 examples [00:09, 846.48 examples/s]8263 examples [00:09, 843.37 examples/s]8350 examples [00:09, 849.25 examples/s]8442 examples [00:09, 867.07 examples/s]8530 examples [00:09, 869.58 examples/s]8619 examples [00:10, 873.09 examples/s]8707 examples [00:10, 874.81 examples/s]8795 examples [00:10, 872.48 examples/s]8883 examples [00:10, 873.67 examples/s]8974 examples [00:10, 882.40 examples/s]9063 examples [00:10, 882.72 examples/s]9152 examples [00:10, 873.94 examples/s]9240 examples [00:10, 864.76 examples/s]9329 examples [00:10, 869.54 examples/s]9416 examples [00:10, 861.37 examples/s]9503 examples [00:11, 862.16 examples/s]9590 examples [00:11, 859.50 examples/s]9680 examples [00:11, 869.17 examples/s]9769 examples [00:11, 873.69 examples/s]9857 examples [00:11, 872.93 examples/s]9945 examples [00:11, 869.53 examples/s]10032 examples [00:11, 815.06 examples/s]10121 examples [00:11, 835.57 examples/s]10210 examples [00:11, 850.64 examples/s]10296 examples [00:11, 853.34 examples/s]10382 examples [00:12, 849.83 examples/s]10471 examples [00:12, 858.60 examples/s]10558 examples [00:12, 861.97 examples/s]10645 examples [00:12, 857.20 examples/s]10731 examples [00:12, 838.14 examples/s]10820 examples [00:12, 851.36 examples/s]10908 examples [00:12, 858.40 examples/s]10998 examples [00:12, 870.30 examples/s]11086 examples [00:12, 868.02 examples/s]11173 examples [00:12, 863.69 examples/s]11260 examples [00:13, 863.95 examples/s]11349 examples [00:13, 869.21 examples/s]11438 examples [00:13, 872.92 examples/s]11530 examples [00:13, 883.83 examples/s]11620 examples [00:13, 887.28 examples/s]11709 examples [00:13, 882.34 examples/s]11798 examples [00:13, 870.38 examples/s]11887 examples [00:13, 873.62 examples/s]11975 examples [00:13, 874.17 examples/s]12064 examples [00:14, 877.28 examples/s]12154 examples [00:14, 882.57 examples/s]12243 examples [00:14, 870.36 examples/s]12331 examples [00:14, 867.54 examples/s]12419 examples [00:14, 868.61 examples/s]12506 examples [00:14, 836.90 examples/s]12596 examples [00:14, 853.04 examples/s]12686 examples [00:14, 864.31 examples/s]12773 examples [00:14, 822.16 examples/s]12860 examples [00:14, 835.81 examples/s]12947 examples [00:15, 842.90 examples/s]13036 examples [00:15, 854.97 examples/s]13122 examples [00:15, 856.04 examples/s]13208 examples [00:15, 847.51 examples/s]13296 examples [00:15, 855.39 examples/s]13382 examples [00:15, 855.57 examples/s]13468 examples [00:15, 848.39 examples/s]13556 examples [00:15, 857.49 examples/s]13644 examples [00:15, 862.63 examples/s]13731 examples [00:15, 861.38 examples/s]13819 examples [00:16, 864.47 examples/s]13906 examples [00:16, 850.48 examples/s]13995 examples [00:16, 860.13 examples/s]14082 examples [00:16, 849.28 examples/s]14172 examples [00:16, 861.59 examples/s]14262 examples [00:16, 871.74 examples/s]14351 examples [00:16, 875.47 examples/s]14440 examples [00:16, 879.56 examples/s]14529 examples [00:16, 882.48 examples/s]14620 examples [00:16, 887.84 examples/s]14709 examples [00:17, 874.12 examples/s]14800 examples [00:17, 883.36 examples/s]14890 examples [00:17, 885.80 examples/s]14982 examples [00:17, 893.76 examples/s]15074 examples [00:17, 899.16 examples/s]15165 examples [00:17, 899.40 examples/s]15255 examples [00:17, 883.26 examples/s]15347 examples [00:17, 892.30 examples/s]15437 examples [00:17, 893.93 examples/s]15527 examples [00:17, 877.12 examples/s]15615 examples [00:18, 870.04 examples/s]15703 examples [00:18, 855.35 examples/s]15789 examples [00:18, 846.93 examples/s]15878 examples [00:18, 858.98 examples/s]15966 examples [00:18, 864.67 examples/s]16055 examples [00:18, 869.72 examples/s]16149 examples [00:18, 887.76 examples/s]16241 examples [00:18, 895.34 examples/s]16331 examples [00:18, 895.85 examples/s]16423 examples [00:19, 902.01 examples/s]16514 examples [00:19, 899.60 examples/s]16605 examples [00:19, 893.48 examples/s]16696 examples [00:19, 896.20 examples/s]16786 examples [00:19, 874.49 examples/s]16874 examples [00:19, 868.73 examples/s]16961 examples [00:19, 868.85 examples/s]17052 examples [00:19, 879.14 examples/s]17140 examples [00:19, 867.31 examples/s]17228 examples [00:19, 870.59 examples/s]17317 examples [00:20, 873.68 examples/s]17405 examples [00:20, 871.89 examples/s]17493 examples [00:20, 860.27 examples/s]17580 examples [00:20, 855.61 examples/s]17671 examples [00:20, 870.04 examples/s]17765 examples [00:20, 887.86 examples/s]17856 examples [00:20, 894.33 examples/s]17946 examples [00:20, 883.20 examples/s]18035 examples [00:20, 876.52 examples/s]18123 examples [00:20, 872.80 examples/s]18211 examples [00:21, 872.95 examples/s]18299 examples [00:21, 870.30 examples/s]18387 examples [00:21, 862.59 examples/s]18477 examples [00:21, 872.19 examples/s]18565 examples [00:21, 848.83 examples/s]18651 examples [00:21, 844.58 examples/s]18736 examples [00:21, 845.25 examples/s]18822 examples [00:21, 849.52 examples/s]18913 examples [00:21, 865.15 examples/s]19007 examples [00:21, 884.34 examples/s]19098 examples [00:22, 890.50 examples/s]19190 examples [00:22, 897.01 examples/s]19280 examples [00:22, 897.69 examples/s]19370 examples [00:22, 896.53 examples/s]19461 examples [00:22, 898.31 examples/s]19553 examples [00:22, 904.54 examples/s]19644 examples [00:22, 891.95 examples/s]19734 examples [00:22, 872.39 examples/s]19822 examples [00:22, 858.22 examples/s]19908 examples [00:23, 856.98 examples/s]19999 examples [00:23, 869.87 examples/s]20087 examples [00:23, 810.84 examples/s]20176 examples [00:23, 831.36 examples/s]20265 examples [00:23, 847.33 examples/s]20353 examples [00:23, 855.76 examples/s]20445 examples [00:23, 871.81 examples/s]20534 examples [00:23, 877.05 examples/s]20622 examples [00:23, 873.31 examples/s]20710 examples [00:23, 871.34 examples/s]20798 examples [00:24, 855.99 examples/s]20888 examples [00:24, 866.57 examples/s]20975 examples [00:24, 850.46 examples/s]21066 examples [00:24, 865.40 examples/s]21153 examples [00:24, 850.59 examples/s]21239 examples [00:24, 850.40 examples/s]21325 examples [00:24, 849.32 examples/s]21411 examples [00:24, 850.46 examples/s]21498 examples [00:24, 854.16 examples/s]21587 examples [00:24, 863.81 examples/s]21676 examples [00:25, 869.04 examples/s]21763 examples [00:25, 858.67 examples/s]21849 examples [00:25, 858.57 examples/s]21939 examples [00:25, 868.31 examples/s]22031 examples [00:25, 881.32 examples/s]22120 examples [00:25, 880.50 examples/s]22209 examples [00:25, 869.76 examples/s]22297 examples [00:25, 858.74 examples/s]22383 examples [00:25, 858.80 examples/s]22469 examples [00:25, 854.50 examples/s]22558 examples [00:26, 862.61 examples/s]22651 examples [00:26, 881.47 examples/s]22740 examples [00:26, 878.08 examples/s]22831 examples [00:26, 884.35 examples/s]22924 examples [00:26, 895.80 examples/s]23014 examples [00:26, 885.48 examples/s]23106 examples [00:26, 893.28 examples/s]23196 examples [00:26, 877.43 examples/s]23289 examples [00:26, 891.10 examples/s]23380 examples [00:27, 896.62 examples/s]23470 examples [00:27, 849.56 examples/s]23562 examples [00:27, 868.81 examples/s]23650 examples [00:27, 869.52 examples/s]23738 examples [00:27, 856.40 examples/s]23827 examples [00:27, 865.02 examples/s]23917 examples [00:27, 872.95 examples/s]24006 examples [00:27, 877.46 examples/s]24096 examples [00:27, 883.01 examples/s]24185 examples [00:27, 858.24 examples/s]24273 examples [00:28, 861.99 examples/s]24360 examples [00:28, 862.85 examples/s]24447 examples [00:28, 857.22 examples/s]24536 examples [00:28, 864.51 examples/s]24624 examples [00:28, 868.70 examples/s]24712 examples [00:28, 870.57 examples/s]24800 examples [00:28, 855.97 examples/s]24891 examples [00:28, 869.22 examples/s]24979 examples [00:28, 862.38 examples/s]25066 examples [00:28, 849.03 examples/s]25152 examples [00:29, 850.41 examples/s]25238 examples [00:29, 842.06 examples/s]25324 examples [00:29, 845.48 examples/s]25409 examples [00:29, 837.11 examples/s]25493 examples [00:29, 822.05 examples/s]25576 examples [00:29, 813.98 examples/s]25658 examples [00:29, 811.58 examples/s]25748 examples [00:29, 835.36 examples/s]25834 examples [00:29, 840.37 examples/s]25926 examples [00:29, 861.32 examples/s]26013 examples [00:30, 858.26 examples/s]26101 examples [00:30, 862.69 examples/s]26188 examples [00:30, 839.21 examples/s]26273 examples [00:30, 819.67 examples/s]26356 examples [00:30, 820.03 examples/s]26442 examples [00:30, 830.88 examples/s]26530 examples [00:30, 843.42 examples/s]26619 examples [00:30, 855.08 examples/s]26705 examples [00:30, 842.49 examples/s]26791 examples [00:31, 846.14 examples/s]26876 examples [00:31, 845.43 examples/s]26961 examples [00:31, 844.31 examples/s]27050 examples [00:31, 854.91 examples/s]27136 examples [00:31, 844.32 examples/s]27229 examples [00:31, 866.83 examples/s]27316 examples [00:31, 863.12 examples/s]27409 examples [00:31, 879.86 examples/s]27498 examples [00:31, 872.28 examples/s]27586 examples [00:31, 854.01 examples/s]27677 examples [00:32, 869.46 examples/s]27768 examples [00:32, 879.83 examples/s]27857 examples [00:32, 866.77 examples/s]27945 examples [00:32, 868.51 examples/s]28034 examples [00:32, 872.82 examples/s]28122 examples [00:32, 871.31 examples/s]28210 examples [00:32, 869.64 examples/s]28298 examples [00:32, 872.48 examples/s]28386 examples [00:32, 867.40 examples/s]28473 examples [00:32, 839.05 examples/s]28560 examples [00:33, 846.24 examples/s]28650 examples [00:33, 860.97 examples/s]28738 examples [00:33, 865.31 examples/s]28826 examples [00:33, 869.21 examples/s]28914 examples [00:33, 845.51 examples/s]29001 examples [00:33, 850.87 examples/s]29088 examples [00:33, 854.68 examples/s]29178 examples [00:33, 867.05 examples/s]29265 examples [00:33, 865.09 examples/s]29352 examples [00:33, 860.18 examples/s]29440 examples [00:34, 859.19 examples/s]29526 examples [00:34, 853.28 examples/s]29614 examples [00:34, 860.49 examples/s]29701 examples [00:34, 810.19 examples/s]29785 examples [00:34, 815.95 examples/s]29868 examples [00:34, 809.18 examples/s]29950 examples [00:34, 779.88 examples/s]30029 examples [00:34, 714.30 examples/s]30103 examples [00:34, 720.46 examples/s]30181 examples [00:35, 736.01 examples/s]30264 examples [00:35, 760.88 examples/s]30348 examples [00:35, 782.03 examples/s]30427 examples [00:35, 779.33 examples/s]30506 examples [00:35, 782.17 examples/s]30586 examples [00:35, 785.00 examples/s]30672 examples [00:35, 805.51 examples/s]30764 examples [00:35, 835.43 examples/s]30849 examples [00:35, 838.72 examples/s]30937 examples [00:35, 848.76 examples/s]31023 examples [00:36, 848.30 examples/s]31111 examples [00:36, 857.23 examples/s]31201 examples [00:36, 867.38 examples/s]31291 examples [00:36, 874.42 examples/s]31379 examples [00:36, 867.66 examples/s]31466 examples [00:36, 866.07 examples/s]31555 examples [00:36, 872.70 examples/s]31643 examples [00:36, 860.10 examples/s]31732 examples [00:36, 868.03 examples/s]31819 examples [00:36, 861.12 examples/s]31906 examples [00:37, 858.34 examples/s]31993 examples [00:37, 860.79 examples/s]32083 examples [00:37, 869.84 examples/s]32175 examples [00:37, 882.22 examples/s]32264 examples [00:37, 883.40 examples/s]32353 examples [00:37, 882.47 examples/s]32442 examples [00:37, 883.15 examples/s]32531 examples [00:37, 871.81 examples/s]32621 examples [00:37, 880.01 examples/s]32710 examples [00:38, 866.95 examples/s]32800 examples [00:38, 874.13 examples/s]32888 examples [00:38, 870.05 examples/s]32978 examples [00:38, 876.15 examples/s]33066 examples [00:38, 876.16 examples/s]33154 examples [00:38, 872.14 examples/s]33243 examples [00:38, 874.55 examples/s]33331 examples [00:38, 870.18 examples/s]33419 examples [00:38, 867.28 examples/s]33510 examples [00:38, 878.11 examples/s]33598 examples [00:39, 857.86 examples/s]33684 examples [00:39, 853.83 examples/s]33775 examples [00:39, 869.91 examples/s]33864 examples [00:39, 875.44 examples/s]33952 examples [00:39, 863.35 examples/s]34039 examples [00:39, 850.17 examples/s]34128 examples [00:39, 861.61 examples/s]34219 examples [00:39, 873.22 examples/s]34309 examples [00:39, 879.72 examples/s]34398 examples [00:39, 881.61 examples/s]34487 examples [00:40, 866.05 examples/s]34576 examples [00:40, 872.14 examples/s]34664 examples [00:40, 868.54 examples/s]34754 examples [00:40, 875.35 examples/s]34842 examples [00:40, 859.07 examples/s]34931 examples [00:40, 867.73 examples/s]35018 examples [00:40, 836.60 examples/s]35102 examples [00:40, 816.49 examples/s]35184 examples [00:40, 803.79 examples/s]35269 examples [00:40, 815.98 examples/s]35357 examples [00:41, 832.01 examples/s]35444 examples [00:41, 841.06 examples/s]35529 examples [00:41, 832.38 examples/s]35613 examples [00:41, 811.07 examples/s]35697 examples [00:41, 819.19 examples/s]35786 examples [00:41, 837.83 examples/s]35876 examples [00:41, 854.25 examples/s]35968 examples [00:41, 870.85 examples/s]36056 examples [00:41, 868.79 examples/s]36146 examples [00:42, 876.87 examples/s]36234 examples [00:42, 863.69 examples/s]36321 examples [00:42, 863.89 examples/s]36408 examples [00:42, 857.41 examples/s]36495 examples [00:42, 859.76 examples/s]36584 examples [00:42, 866.79 examples/s]36672 examples [00:42, 869.41 examples/s]36762 examples [00:42, 877.23 examples/s]36850 examples [00:42, 876.78 examples/s]36938 examples [00:42, 862.02 examples/s]37028 examples [00:43, 870.95 examples/s]37116 examples [00:43, 873.13 examples/s]37204 examples [00:43, 864.55 examples/s]37291 examples [00:43, 865.83 examples/s]37378 examples [00:43, 835.53 examples/s]37470 examples [00:43, 859.09 examples/s]37560 examples [00:43, 870.39 examples/s]37650 examples [00:43, 878.08 examples/s]37739 examples [00:43, 880.70 examples/s]37828 examples [00:43, 876.77 examples/s]37916 examples [00:44, 869.90 examples/s]38007 examples [00:44, 879.10 examples/s]38099 examples [00:44, 889.07 examples/s]38189 examples [00:44, 891.63 examples/s]38279 examples [00:44, 881.11 examples/s]38368 examples [00:44, 861.22 examples/s]38457 examples [00:44, 866.56 examples/s]38546 examples [00:44, 871.28 examples/s]38634 examples [00:44, 873.87 examples/s]38724 examples [00:44, 880.81 examples/s]38813 examples [00:45, 876.75 examples/s]38901 examples [00:45, 871.31 examples/s]38989 examples [00:45, 854.00 examples/s]39078 examples [00:45, 858.52 examples/s]39170 examples [00:45, 875.45 examples/s]39258 examples [00:45, 872.75 examples/s]39351 examples [00:45, 888.45 examples/s]39440 examples [00:45, 888.26 examples/s]39529 examples [00:45, 887.90 examples/s]39618 examples [00:45, 865.54 examples/s]39705 examples [00:46, 857.34 examples/s]39795 examples [00:46, 867.81 examples/s]39883 examples [00:46, 868.71 examples/s]39970 examples [00:46, 866.10 examples/s]40057 examples [00:46, 805.14 examples/s]40146 examples [00:46, 827.12 examples/s]40238 examples [00:46, 851.93 examples/s]40330 examples [00:46, 869.79 examples/s]40420 examples [00:46, 876.54 examples/s]40510 examples [00:47, 881.31 examples/s]40599 examples [00:47, 879.14 examples/s]40688 examples [00:47, 872.54 examples/s]40776 examples [00:47, 833.35 examples/s]40863 examples [00:47, 842.42 examples/s]40948 examples [00:47, 844.45 examples/s]41035 examples [00:47, 850.97 examples/s]41125 examples [00:47, 862.83 examples/s]41214 examples [00:47, 868.41 examples/s]41301 examples [00:47, 855.42 examples/s]41388 examples [00:48, 858.24 examples/s]41474 examples [00:48, 832.06 examples/s]41560 examples [00:48, 839.25 examples/s]41650 examples [00:48, 854.74 examples/s]41736 examples [00:48, 854.90 examples/s]41826 examples [00:48, 867.07 examples/s]41915 examples [00:48, 873.62 examples/s]42004 examples [00:48, 878.23 examples/s]42093 examples [00:48, 878.21 examples/s]42181 examples [00:48, 876.61 examples/s]42269 examples [00:49, 872.76 examples/s]42357 examples [00:49, 872.07 examples/s]42445 examples [00:49, 873.88 examples/s]42533 examples [00:49, 868.08 examples/s]42622 examples [00:49, 873.04 examples/s]42711 examples [00:49, 876.41 examples/s]42799 examples [00:49, 869.31 examples/s]42891 examples [00:49, 882.47 examples/s]42980 examples [00:49, 876.72 examples/s]43068 examples [00:49, 867.34 examples/s]43159 examples [00:50, 877.84 examples/s]43250 examples [00:50, 886.61 examples/s]43342 examples [00:50, 896.08 examples/s]43433 examples [00:50, 899.21 examples/s]43524 examples [00:50, 899.58 examples/s]43616 examples [00:50, 903.62 examples/s]43707 examples [00:50, 893.37 examples/s]43798 examples [00:50, 897.13 examples/s]43888 examples [00:50, 888.98 examples/s]43977 examples [00:51, 884.84 examples/s]44066 examples [00:51, 873.61 examples/s]44154 examples [00:51, 870.43 examples/s]44242 examples [00:51, 865.38 examples/s]44329 examples [00:51, 865.98 examples/s]44417 examples [00:51, 868.63 examples/s]44504 examples [00:51, 865.89 examples/s]44592 examples [00:51, 869.60 examples/s]44680 examples [00:51, 871.76 examples/s]44768 examples [00:51, 873.30 examples/s]44856 examples [00:52, 872.01 examples/s]44945 examples [00:52, 876.48 examples/s]45033 examples [00:52, 851.86 examples/s]45122 examples [00:52, 860.22 examples/s]45211 examples [00:52, 867.66 examples/s]45302 examples [00:52, 876.99 examples/s]45390 examples [00:52, 876.98 examples/s]45478 examples [00:52, 868.86 examples/s]45567 examples [00:52, 874.32 examples/s]45655 examples [00:52, 873.34 examples/s]45743 examples [00:53, 871.82 examples/s]45832 examples [00:53, 875.80 examples/s]45920 examples [00:53, 865.89 examples/s]46009 examples [00:53, 871.27 examples/s]46099 examples [00:53, 878.65 examples/s]46189 examples [00:53, 882.39 examples/s]46280 examples [00:53, 890.06 examples/s]46370 examples [00:53, 887.67 examples/s]46460 examples [00:53, 890.47 examples/s]46550 examples [00:53, 888.06 examples/s]46639 examples [00:54, 882.38 examples/s]46728 examples [00:54, 875.28 examples/s]46816 examples [00:54, 872.88 examples/s]46906 examples [00:54, 879.51 examples/s]46995 examples [00:54, 882.49 examples/s]47086 examples [00:54, 890.23 examples/s]47179 examples [00:54, 899.32 examples/s]47270 examples [00:54, 901.31 examples/s]47361 examples [00:54, 867.47 examples/s]47449 examples [00:54, 865.70 examples/s]47536 examples [00:55, 848.18 examples/s]47622 examples [00:55, 850.13 examples/s]47714 examples [00:55, 867.82 examples/s]47801 examples [00:55, 861.84 examples/s]47888 examples [00:55, 861.87 examples/s]47975 examples [00:55, 863.80 examples/s]48062 examples [00:55, 865.00 examples/s]48149 examples [00:55, 865.01 examples/s]48239 examples [00:55, 873.40 examples/s]48327 examples [00:56, 859.84 examples/s]48415 examples [00:56, 863.22 examples/s]48502 examples [00:56, 837.62 examples/s]48590 examples [00:56, 847.37 examples/s]48675 examples [00:56, 815.56 examples/s]48757 examples [00:56, 789.21 examples/s]48844 examples [00:56, 810.06 examples/s]48932 examples [00:56, 829.39 examples/s]49020 examples [00:56, 842.04 examples/s]49109 examples [00:56, 853.77 examples/s]49196 examples [00:57, 856.12 examples/s]49287 examples [00:57, 869.98 examples/s]49375 examples [00:57, 867.52 examples/s]49465 examples [00:57, 874.54 examples/s]49553 examples [00:57, 831.77 examples/s]49639 examples [00:57, 838.49 examples/s]49724 examples [00:57, 836.46 examples/s]49815 examples [00:57, 856.77 examples/s]49904 examples [00:57, 865.38 examples/s]49994 examples [00:57, 874.65 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|█         | 5187/50000 [00:00<00:00, 51865.35 examples/s] 30%|███       | 15026/50000 [00:00<00:00, 60438.41 examples/s] 50%|█████     | 25054/50000 [00:00<00:00, 68616.37 examples/s] 70%|██████▉   | 34913/50000 [00:00<00:00, 75502.10 examples/s] 89%|████████▉ | 44437/50000 [00:00<00:00, 80506.74 examples/s]                                                               0 examples [00:00, ? examples/s]57 examples [00:00, 564.03 examples/s]147 examples [00:00, 633.92 examples/s]237 examples [00:00, 694.40 examples/s]327 examples [00:00, 744.34 examples/s]417 examples [00:00, 783.93 examples/s]507 examples [00:00, 815.10 examples/s]598 examples [00:00, 839.77 examples/s]683 examples [00:00, 841.20 examples/s]775 examples [00:00, 861.02 examples/s]865 examples [00:01, 870.44 examples/s]957 examples [00:01, 883.59 examples/s]1047 examples [00:01, 887.31 examples/s]1137 examples [00:01, 889.52 examples/s]1228 examples [00:01, 894.87 examples/s]1321 examples [00:01, 904.08 examples/s]1412 examples [00:01, 891.51 examples/s]1502 examples [00:01, 892.41 examples/s]1595 examples [00:01, 902.56 examples/s]1686 examples [00:01, 882.69 examples/s]1777 examples [00:02, 888.13 examples/s]1866 examples [00:02, 870.41 examples/s]1956 examples [00:02, 878.63 examples/s]2044 examples [00:02, 878.29 examples/s]2132 examples [00:02, 878.78 examples/s]2222 examples [00:02, 884.76 examples/s]2311 examples [00:02, 879.96 examples/s]2402 examples [00:02, 885.82 examples/s]2493 examples [00:02, 890.20 examples/s]2584 examples [00:02, 893.87 examples/s]2674 examples [00:03, 874.44 examples/s]2762 examples [00:03, 864.60 examples/s]2854 examples [00:03, 877.58 examples/s]2946 examples [00:03, 887.38 examples/s]3038 examples [00:03, 894.45 examples/s]3129 examples [00:03, 898.61 examples/s]3219 examples [00:03, 896.10 examples/s]3309 examples [00:03, 887.97 examples/s]3399 examples [00:03, 889.21 examples/s]3489 examples [00:03, 892.06 examples/s]3581 examples [00:04, 899.99 examples/s]3672 examples [00:04, 900.73 examples/s]3764 examples [00:04, 903.22 examples/s]3855 examples [00:04, 887.26 examples/s]3944 examples [00:04, 875.92 examples/s]4035 examples [00:04, 884.59 examples/s]4125 examples [00:04, 888.71 examples/s]4215 examples [00:04, 891.54 examples/s]4305 examples [00:04, 889.14 examples/s]4394 examples [00:04, 879.06 examples/s]4485 examples [00:05, 885.68 examples/s]4574 examples [00:05, 886.36 examples/s]4663 examples [00:05, 885.86 examples/s]4753 examples [00:05, 888.26 examples/s]4843 examples [00:05, 890.50 examples/s]4933 examples [00:05, 891.24 examples/s]5023 examples [00:05, 872.98 examples/s]5113 examples [00:05, 878.03 examples/s]5201 examples [00:05, 865.51 examples/s]5295 examples [00:05, 885.49 examples/s]5384 examples [00:06, 885.29 examples/s]5473 examples [00:06, 876.98 examples/s]5567 examples [00:06, 892.65 examples/s]5657 examples [00:06, 862.05 examples/s]5748 examples [00:06, 874.53 examples/s]5836 examples [00:06, 874.69 examples/s]5927 examples [00:06, 883.46 examples/s]6016 examples [00:06, 884.58 examples/s]6105 examples [00:06, 867.84 examples/s]6192 examples [00:07, 863.92 examples/s]6279 examples [00:07, 855.75 examples/s]6369 examples [00:07, 867.57 examples/s]6456 examples [00:07, 863.41 examples/s]6548 examples [00:07, 879.43 examples/s]6640 examples [00:07, 889.09 examples/s]6730 examples [00:07, 871.17 examples/s]6818 examples [00:07, 867.51 examples/s]6908 examples [00:07, 876.19 examples/s]6996 examples [00:07, 858.60 examples/s]7086 examples [00:08, 868.00 examples/s]7177 examples [00:08, 878.74 examples/s]7265 examples [00:08, 856.50 examples/s]7355 examples [00:08, 867.82 examples/s]7446 examples [00:08, 877.67 examples/s]7537 examples [00:08, 885.06 examples/s]7628 examples [00:08, 890.13 examples/s]7718 examples [00:08, 892.12 examples/s]7808 examples [00:08, 872.66 examples/s]7896 examples [00:08, 851.68 examples/s]7982 examples [00:09, 851.69 examples/s]8069 examples [00:09, 854.84 examples/s]8155 examples [00:09, 852.21 examples/s]8242 examples [00:09, 856.01 examples/s]8330 examples [00:09, 861.00 examples/s]8417 examples [00:09, 863.39 examples/s]8504 examples [00:09, 853.51 examples/s]8593 examples [00:09, 862.62 examples/s]8682 examples [00:09, 869.79 examples/s]8770 examples [00:10, 849.48 examples/s]8862 examples [00:10, 867.82 examples/s]8953 examples [00:10, 879.90 examples/s]9044 examples [00:10, 887.79 examples/s]9133 examples [00:10, 882.12 examples/s]9223 examples [00:10, 885.62 examples/s]9312 examples [00:10, 884.70 examples/s]9401 examples [00:10, 879.12 examples/s]9492 examples [00:10, 886.64 examples/s]9581 examples [00:10, 881.72 examples/s]9670 examples [00:11, 873.47 examples/s]9758 examples [00:11, 868.64 examples/s]9845 examples [00:11, 847.52 examples/s]9933 examples [00:11, 853.81 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 95%|█████████▍| 9481/10000 [00:00<00:00, 94801.28 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKGCCY0/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteKGCCY0/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4827bf2158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4827bf2158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4827bf2158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f48735d00f0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f48930e1f98>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f4827bf2158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f4827bf2158> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f4827bf2158> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f48066a65f8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f4825fda390>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.06 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  9.06 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  9.06 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.68 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  8.68 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.66 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.93 file/s]2020-08-11 12:09:46.102222: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-11 12:09:46.106582: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294680000 Hz
2020-08-11 12:09:46.106856: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5618333f33b0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-11 12:09:46.106875: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 49152/9912422 [00:00<00:20, 475031.46it/s] 86%|████████▌ | 8486912/9912422 [00:00<00:02, 676969.52it/s]9920512it [00:00, 45194272.05it/s]                           
0it [00:00, ?it/s]32768it [00:00, 508543.45it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 1004792.75it/s]1654784it [00:00, 12367744.94it/s]                           
0it [00:00, ?it/s]8192it [00:00, 246057.66it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7fead9cb2598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7fead9cb2598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7feb44964400> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7feb44964400> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7feb63b80ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7feb63b80ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feaf1ca00d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feaf1ca00d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feaf1ca00d0> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 41%|████      | 4030464/9912422 [00:00<00:00, 39584952.04it/s]9920512it [00:00, 30099655.35it/s]                             
0it [00:00, ?it/s]32768it [00:00, 565983.70it/s]
0it [00:00, ?it/s] 10%|█         | 172032/1648877 [00:00<00:00, 1707372.75it/s]1654784it [00:00, 13044625.91it/s]                           
0it [00:00, ?it/s]8192it [00:00, 154306.51it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feaefcff4e0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feaefe51358>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7feaf1c97d08> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7feaf1c97d08> 

  function with postional parmater data_info <function tf_dataset_download at 0x7feaf1c97d08> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:23,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:22,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:22,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:21,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:21,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:20,  1.94 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:56,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:56,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:56,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:55,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:54,  2.72 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:38,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:38,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:38,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:37,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:37,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:37,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:37,  3.82 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:26,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:25,  5.33 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:18,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:17,  7.37 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12, 10.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:12, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:12, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:12, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:12, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:12, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:12, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:08, 13.52 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.81 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:06, 17.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 17.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 17.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 17.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:06, 17.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:06, 17.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 17.81 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 22.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 22.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 22.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 22.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 22.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 22.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 22.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 22.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.58 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 28.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 28.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 28.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 28.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 28.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 28.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 28.58 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 34.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 40.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 40.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 40.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 40.19 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 45.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 45.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 50.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 50.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.33 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 54.33 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 54.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 54.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 54.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 54.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 54.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 54.33 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 56.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 56.91 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.89 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 62.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 62.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 62.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 62.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 62.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 62.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 62.89 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 64.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 64.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 64.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 64.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 64.31 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.98 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 64.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 64.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 64.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 64.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 64.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 64.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 64.98 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 65.37 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 67.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.34 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 68.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 68.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 68.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 68.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.91s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.91s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.34 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.91s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 68.34 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.40s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.91s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 68.34 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.40s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.40s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.02 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.40s/ url]
0 examples [00:00, ? examples/s]2020-08-19 00:07:27.990396: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-19 00:07:28.007043: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-08-19 00:07:28.007251: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a6927fcfd0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-19 00:07:28.007271: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
26 examples [00:00, 258.25 examples/s]118 examples [00:00, 329.23 examples/s]211 examples [00:00, 408.29 examples/s]309 examples [00:00, 494.69 examples/s]407 examples [00:00, 580.07 examples/s]497 examples [00:00, 649.10 examples/s]599 examples [00:00, 727.14 examples/s]688 examples [00:00, 767.74 examples/s]781 examples [00:00, 808.50 examples/s]879 examples [00:01, 853.00 examples/s]971 examples [00:01, 869.57 examples/s]1063 examples [00:01, 850.66 examples/s]1156 examples [00:01, 872.89 examples/s]1252 examples [00:01, 895.08 examples/s]1346 examples [00:01, 905.81 examples/s]1444 examples [00:01, 924.73 examples/s]1538 examples [00:01, 915.62 examples/s]1631 examples [00:01, 885.72 examples/s]1724 examples [00:01, 895.66 examples/s]1817 examples [00:02, 902.15 examples/s]1908 examples [00:02, 873.00 examples/s]2000 examples [00:02, 886.08 examples/s]2094 examples [00:02, 900.10 examples/s]2187 examples [00:02, 907.38 examples/s]2278 examples [00:02, 902.90 examples/s]2373 examples [00:02, 914.06 examples/s]2467 examples [00:02, 921.19 examples/s]2560 examples [00:02, 923.30 examples/s]2656 examples [00:02, 931.32 examples/s]2751 examples [00:03, 935.19 examples/s]2845 examples [00:03, 924.47 examples/s]2941 examples [00:03, 932.15 examples/s]3036 examples [00:03, 936.04 examples/s]3130 examples [00:03, 934.61 examples/s]3226 examples [00:03, 940.98 examples/s]3321 examples [00:03, 935.76 examples/s]3420 examples [00:03, 951.28 examples/s]3517 examples [00:03, 954.15 examples/s]3616 examples [00:03, 962.46 examples/s]3713 examples [00:04, 961.49 examples/s]3810 examples [00:04, 945.40 examples/s]3907 examples [00:04, 950.40 examples/s]4003 examples [00:04, 942.41 examples/s]4099 examples [00:04, 946.30 examples/s]4196 examples [00:04, 952.40 examples/s]4292 examples [00:04, 954.40 examples/s]4389 examples [00:04, 956.56 examples/s]4485 examples [00:04, 936.45 examples/s]4581 examples [00:04, 941.73 examples/s]4676 examples [00:05, 934.06 examples/s]4774 examples [00:05, 946.47 examples/s]4874 examples [00:05, 959.09 examples/s]4971 examples [00:05, 950.94 examples/s]5069 examples [00:05, 952.44 examples/s]5165 examples [00:05, 952.84 examples/s]5261 examples [00:05, 952.85 examples/s]5357 examples [00:05, 949.56 examples/s]5452 examples [00:05, 942.02 examples/s]5547 examples [00:06, 944.34 examples/s]5642 examples [00:06, 938.89 examples/s]5737 examples [00:06, 941.71 examples/s]5834 examples [00:06, 947.55 examples/s]5929 examples [00:06, 938.36 examples/s]6023 examples [00:06, 919.48 examples/s]6116 examples [00:06, 912.51 examples/s]6208 examples [00:06, 912.99 examples/s]6305 examples [00:06, 928.60 examples/s]6398 examples [00:06, 922.24 examples/s]6497 examples [00:07, 939.70 examples/s]6592 examples [00:07, 941.16 examples/s]6691 examples [00:07, 953.75 examples/s]6787 examples [00:07, 918.15 examples/s]6883 examples [00:07, 928.35 examples/s]6982 examples [00:07, 945.40 examples/s]7078 examples [00:07, 948.01 examples/s]7174 examples [00:07, 949.71 examples/s]7270 examples [00:07, 948.20 examples/s]7365 examples [00:07, 937.46 examples/s]7459 examples [00:08, 932.03 examples/s]7553 examples [00:08, 925.29 examples/s]7651 examples [00:08, 940.36 examples/s]7746 examples [00:08, 943.11 examples/s]7842 examples [00:08, 946.05 examples/s]7940 examples [00:08, 955.94 examples/s]8036 examples [00:08, 946.28 examples/s]8131 examples [00:08, 940.05 examples/s]8226 examples [00:08, 941.04 examples/s]8321 examples [00:08, 938.87 examples/s]8415 examples [00:09, 928.07 examples/s]8508 examples [00:09, 903.22 examples/s]8602 examples [00:09, 913.55 examples/s]8694 examples [00:09, 911.05 examples/s]8786 examples [00:09, 913.01 examples/s]8879 examples [00:09, 917.14 examples/s]8971 examples [00:09, 891.79 examples/s]9069 examples [00:09, 914.49 examples/s]9164 examples [00:09, 922.48 examples/s]9261 examples [00:10, 934.35 examples/s]9355 examples [00:10, 935.14 examples/s]9450 examples [00:10, 939.06 examples/s]9545 examples [00:10, 933.34 examples/s]9642 examples [00:10, 942.55 examples/s]9738 examples [00:10, 945.71 examples/s]9835 examples [00:10, 950.85 examples/s]9931 examples [00:10, 944.78 examples/s]10026 examples [00:10, 897.09 examples/s]10117 examples [00:10, 900.54 examples/s]10210 examples [00:11, 907.89 examples/s]10302 examples [00:11, 907.36 examples/s]10394 examples [00:11, 910.33 examples/s]10486 examples [00:11, 898.13 examples/s]10582 examples [00:11, 913.95 examples/s]10674 examples [00:11, 913.33 examples/s]10771 examples [00:11, 927.35 examples/s]10867 examples [00:11, 936.08 examples/s]10962 examples [00:11, 938.92 examples/s]11059 examples [00:11, 947.05 examples/s]11154 examples [00:12, 947.66 examples/s]11249 examples [00:12, 940.35 examples/s]11344 examples [00:12, 910.68 examples/s]11440 examples [00:12, 924.25 examples/s]11536 examples [00:12, 932.28 examples/s]11631 examples [00:12, 936.23 examples/s]11725 examples [00:12, 924.93 examples/s]11822 examples [00:12, 937.20 examples/s]11917 examples [00:12, 939.73 examples/s]12012 examples [00:12, 922.59 examples/s]12109 examples [00:13, 933.71 examples/s]12203 examples [00:13, 921.91 examples/s]12305 examples [00:13, 947.64 examples/s]12402 examples [00:13, 952.98 examples/s]12500 examples [00:13, 958.14 examples/s]12596 examples [00:13, 956.67 examples/s]12695 examples [00:13, 965.07 examples/s]12792 examples [00:13, 963.58 examples/s]12889 examples [00:13, 964.77 examples/s]12990 examples [00:13, 975.06 examples/s]13088 examples [00:14, 962.91 examples/s]13185 examples [00:14, 954.71 examples/s]13281 examples [00:14, 948.54 examples/s]13378 examples [00:14, 952.68 examples/s]13476 examples [00:14, 959.62 examples/s]13573 examples [00:14, 951.07 examples/s]13669 examples [00:14, 942.60 examples/s]13764 examples [00:14, 931.47 examples/s]13859 examples [00:14, 936.01 examples/s]13953 examples [00:15, 924.41 examples/s]14046 examples [00:15, 925.76 examples/s]14142 examples [00:15, 932.88 examples/s]14238 examples [00:15, 939.34 examples/s]14333 examples [00:15, 942.30 examples/s]14428 examples [00:15, 932.65 examples/s]14527 examples [00:15, 947.27 examples/s]14622 examples [00:15, 933.72 examples/s]14718 examples [00:15, 940.80 examples/s]14817 examples [00:15, 952.86 examples/s]14916 examples [00:16, 962.80 examples/s]15013 examples [00:16, 934.68 examples/s]15107 examples [00:16, 925.88 examples/s]15200 examples [00:16, 919.33 examples/s]15293 examples [00:16, 914.37 examples/s]15385 examples [00:16, 909.35 examples/s]15477 examples [00:16, 902.16 examples/s]15570 examples [00:16, 908.78 examples/s]15664 examples [00:16, 915.57 examples/s]15761 examples [00:16, 928.71 examples/s]15854 examples [00:17, 901.72 examples/s]15945 examples [00:17, 898.55 examples/s]16036 examples [00:17, 889.55 examples/s]16126 examples [00:17, 889.24 examples/s]16219 examples [00:17, 900.65 examples/s]16310 examples [00:17, 900.88 examples/s]16401 examples [00:17, 897.41 examples/s]16494 examples [00:17, 904.53 examples/s]16587 examples [00:17, 912.02 examples/s]16684 examples [00:17, 926.65 examples/s]16777 examples [00:18, 921.37 examples/s]16870 examples [00:18, 881.63 examples/s]16963 examples [00:18, 895.06 examples/s]17060 examples [00:18, 913.85 examples/s]17157 examples [00:18, 929.64 examples/s]17253 examples [00:18, 937.58 examples/s]17347 examples [00:18, 922.73 examples/s]17440 examples [00:18, 920.45 examples/s]17533 examples [00:18, 922.47 examples/s]17626 examples [00:19, 919.04 examples/s]17718 examples [00:19, 918.13 examples/s]17812 examples [00:19, 923.17 examples/s]17905 examples [00:19, 900.96 examples/s]17996 examples [00:19, 889.31 examples/s]18086 examples [00:19, 890.25 examples/s]18182 examples [00:19, 908.58 examples/s]18275 examples [00:19, 914.74 examples/s]18367 examples [00:19, 906.69 examples/s]18462 examples [00:19, 916.75 examples/s]18554 examples [00:20, 914.52 examples/s]18646 examples [00:20, 911.97 examples/s]18743 examples [00:20, 926.89 examples/s]18841 examples [00:20, 939.76 examples/s]18936 examples [00:20, 925.63 examples/s]19029 examples [00:20, 921.11 examples/s]19122 examples [00:20, 880.98 examples/s]19213 examples [00:20, 889.46 examples/s]19303 examples [00:20, 886.34 examples/s]19396 examples [00:20, 896.35 examples/s]19486 examples [00:21, 897.19 examples/s]19576 examples [00:21, 897.60 examples/s]19668 examples [00:21, 897.01 examples/s]19763 examples [00:21, 910.12 examples/s]19858 examples [00:21, 921.26 examples/s]19952 examples [00:21, 924.75 examples/s]20045 examples [00:21, 887.86 examples/s]20138 examples [00:21, 898.97 examples/s]20230 examples [00:21, 903.55 examples/s]20323 examples [00:21, 909.77 examples/s]20419 examples [00:22, 923.46 examples/s]20512 examples [00:22, 921.63 examples/s]20608 examples [00:22, 931.12 examples/s]20704 examples [00:22, 939.11 examples/s]20798 examples [00:22, 859.35 examples/s]20888 examples [00:22, 870.82 examples/s]20979 examples [00:22, 881.58 examples/s]21072 examples [00:22, 895.17 examples/s]21166 examples [00:22, 907.43 examples/s]21258 examples [00:23, 904.38 examples/s]21351 examples [00:23, 911.82 examples/s]21443 examples [00:23, 899.76 examples/s]21536 examples [00:23, 907.86 examples/s]21629 examples [00:23, 913.77 examples/s]21724 examples [00:23, 921.61 examples/s]21818 examples [00:23, 925.43 examples/s]21915 examples [00:23, 938.23 examples/s]22010 examples [00:23, 941.24 examples/s]22107 examples [00:23, 948.02 examples/s]22202 examples [00:24, 929.76 examples/s]22298 examples [00:24, 937.46 examples/s]22393 examples [00:24, 939.62 examples/s]22489 examples [00:24, 944.95 examples/s]22588 examples [00:24, 955.27 examples/s]22686 examples [00:24, 961.57 examples/s]22783 examples [00:24, 932.49 examples/s]22877 examples [00:24, 928.00 examples/s]22971 examples [00:24, 931.47 examples/s]23066 examples [00:24, 936.44 examples/s]23162 examples [00:25, 941.71 examples/s]23257 examples [00:25, 941.55 examples/s]23352 examples [00:25, 943.99 examples/s]23447 examples [00:25, 937.66 examples/s]23541 examples [00:25, 920.43 examples/s]23635 examples [00:25, 925.58 examples/s]23729 examples [00:25, 927.36 examples/s]23822 examples [00:25, 920.85 examples/s]23917 examples [00:25, 927.30 examples/s]24012 examples [00:25, 931.16 examples/s]24106 examples [00:26, 931.03 examples/s]24200 examples [00:26, 931.94 examples/s]24294 examples [00:26, 926.60 examples/s]24388 examples [00:26, 929.95 examples/s]24482 examples [00:26, 931.66 examples/s]24577 examples [00:26, 934.56 examples/s]24671 examples [00:26, 915.68 examples/s]24763 examples [00:26, 909.70 examples/s]24855 examples [00:26, 911.47 examples/s]24947 examples [00:26, 895.98 examples/s]25039 examples [00:27, 901.97 examples/s]25133 examples [00:27, 912.30 examples/s]25227 examples [00:27, 920.38 examples/s]25320 examples [00:27, 890.06 examples/s]25416 examples [00:27, 907.63 examples/s]25508 examples [00:27, 869.67 examples/s]25600 examples [00:27, 883.77 examples/s]25696 examples [00:27, 904.89 examples/s]25788 examples [00:27, 900.70 examples/s]25883 examples [00:28, 914.05 examples/s]25977 examples [00:28, 919.32 examples/s]26070 examples [00:28, 900.26 examples/s]26163 examples [00:28, 907.40 examples/s]26254 examples [00:28, 905.41 examples/s]26347 examples [00:28, 911.34 examples/s]26440 examples [00:28, 914.85 examples/s]26532 examples [00:28, 906.85 examples/s]26624 examples [00:28, 907.92 examples/s]26715 examples [00:28, 902.02 examples/s]26806 examples [00:29, 892.01 examples/s]26902 examples [00:29, 909.84 examples/s]26997 examples [00:29, 919.31 examples/s]27090 examples [00:29, 920.85 examples/s]27183 examples [00:29, 911.53 examples/s]27279 examples [00:29, 925.50 examples/s]27375 examples [00:29, 932.90 examples/s]27470 examples [00:29, 936.12 examples/s]27565 examples [00:29, 932.08 examples/s]27663 examples [00:29, 944.53 examples/s]27761 examples [00:30, 954.27 examples/s]27859 examples [00:30, 959.10 examples/s]27955 examples [00:30, 950.87 examples/s]28051 examples [00:30, 931.53 examples/s]28145 examples [00:30, 926.88 examples/s]28240 examples [00:30, 931.16 examples/s]28334 examples [00:30, 914.24 examples/s]28426 examples [00:30, 910.74 examples/s]28518 examples [00:30, 911.77 examples/s]28610 examples [00:30, 912.45 examples/s]28702 examples [00:31, 906.12 examples/s]28795 examples [00:31, 910.92 examples/s]28887 examples [00:31, 913.42 examples/s]28979 examples [00:31, 877.71 examples/s]29068 examples [00:31, 880.27 examples/s]29161 examples [00:31, 893.11 examples/s]29251 examples [00:31, 895.10 examples/s]29344 examples [00:31, 903.90 examples/s]29439 examples [00:31, 915.34 examples/s]29535 examples [00:32, 928.18 examples/s]29628 examples [00:32, 927.20 examples/s]29722 examples [00:32, 928.45 examples/s]29817 examples [00:32, 934.38 examples/s]29914 examples [00:32, 940.81 examples/s]30009 examples [00:32, 890.18 examples/s]30105 examples [00:32, 908.06 examples/s]30197 examples [00:32, 904.15 examples/s]30288 examples [00:32, 899.88 examples/s]30384 examples [00:32, 914.86 examples/s]30480 examples [00:33, 926.50 examples/s]30573 examples [00:33, 926.71 examples/s]30668 examples [00:33, 931.05 examples/s]30764 examples [00:33, 937.39 examples/s]30858 examples [00:33, 931.30 examples/s]30955 examples [00:33, 940.84 examples/s]31055 examples [00:33, 957.02 examples/s]31151 examples [00:33, 923.31 examples/s]31244 examples [00:33, 917.22 examples/s]31340 examples [00:33, 927.78 examples/s]31436 examples [00:34, 934.85 examples/s]31532 examples [00:34, 940.95 examples/s]31628 examples [00:34, 946.53 examples/s]31723 examples [00:34, 928.85 examples/s]31817 examples [00:34, 916.59 examples/s]31916 examples [00:34, 934.69 examples/s]32011 examples [00:34, 937.68 examples/s]32106 examples [00:34, 938.65 examples/s]32201 examples [00:34, 940.11 examples/s]32296 examples [00:34, 910.07 examples/s]32391 examples [00:35, 920.07 examples/s]32487 examples [00:35, 929.78 examples/s]32581 examples [00:35, 930.37 examples/s]32675 examples [00:35, 927.68 examples/s]32771 examples [00:35, 934.93 examples/s]32866 examples [00:35, 937.75 examples/s]32964 examples [00:35, 948.86 examples/s]33059 examples [00:35, 941.25 examples/s]33154 examples [00:35, 933.96 examples/s]33250 examples [00:36, 939.23 examples/s]33347 examples [00:36, 946.92 examples/s]33444 examples [00:36, 952.23 examples/s]33540 examples [00:36, 949.84 examples/s]33636 examples [00:36, 948.98 examples/s]33733 examples [00:36, 954.18 examples/s]33834 examples [00:36, 967.70 examples/s]33931 examples [00:36, 929.57 examples/s]34025 examples [00:36, 907.18 examples/s]34120 examples [00:36, 919.19 examples/s]34216 examples [00:37, 930.20 examples/s]34312 examples [00:37, 936.18 examples/s]34409 examples [00:37, 943.64 examples/s]34508 examples [00:37, 956.02 examples/s]34604 examples [00:37, 940.45 examples/s]34699 examples [00:37, 918.79 examples/s]34798 examples [00:37, 938.13 examples/s]34896 examples [00:37, 949.04 examples/s]34993 examples [00:37, 952.85 examples/s]35089 examples [00:37, 933.03 examples/s]35183 examples [00:38, 929.28 examples/s]35278 examples [00:38, 934.53 examples/s]35372 examples [00:38, 909.58 examples/s]35464 examples [00:38, 902.66 examples/s]35557 examples [00:38, 909.75 examples/s]35649 examples [00:38, 912.03 examples/s]35743 examples [00:38, 917.71 examples/s]35835 examples [00:38, 917.34 examples/s]35929 examples [00:38, 922.62 examples/s]36022 examples [00:38, 923.32 examples/s]36115 examples [00:39, 922.31 examples/s]36208 examples [00:39, 922.54 examples/s]36306 examples [00:39, 936.63 examples/s]36400 examples [00:39, 936.81 examples/s]36494 examples [00:39, 915.29 examples/s]36590 examples [00:39, 925.24 examples/s]36683 examples [00:39, 912.40 examples/s]36776 examples [00:39, 914.79 examples/s]36868 examples [00:39, 896.99 examples/s]36964 examples [00:40, 914.81 examples/s]37056 examples [00:40, 896.42 examples/s]37152 examples [00:40, 912.41 examples/s]37244 examples [00:40, 909.13 examples/s]37336 examples [00:40, 909.51 examples/s]37428 examples [00:40, 866.70 examples/s]37525 examples [00:40, 892.52 examples/s]37615 examples [00:40, 891.04 examples/s]37706 examples [00:40, 894.14 examples/s]37796 examples [00:40, 867.31 examples/s]37890 examples [00:41, 887.62 examples/s]37986 examples [00:41, 906.19 examples/s]38077 examples [00:41, 905.96 examples/s]38171 examples [00:41, 913.67 examples/s]38263 examples [00:41, 901.66 examples/s]38354 examples [00:41, 902.76 examples/s]38445 examples [00:41, 890.82 examples/s]38536 examples [00:41, 894.80 examples/s]38633 examples [00:41, 913.75 examples/s]38731 examples [00:41, 931.79 examples/s]38831 examples [00:42, 949.42 examples/s]38927 examples [00:42, 941.47 examples/s]39023 examples [00:42, 946.28 examples/s]39118 examples [00:42, 945.63 examples/s]39213 examples [00:42, 945.28 examples/s]39308 examples [00:42, 938.85 examples/s]39402 examples [00:42, 911.90 examples/s]39495 examples [00:42, 914.69 examples/s]39591 examples [00:42, 927.44 examples/s]39686 examples [00:42, 931.82 examples/s]39780 examples [00:43, 911.82 examples/s]39877 examples [00:43, 926.24 examples/s]39977 examples [00:43, 943.79 examples/s]40072 examples [00:43, 884.08 examples/s]40162 examples [00:43, 864.68 examples/s]40258 examples [00:43, 890.05 examples/s]40354 examples [00:43, 908.65 examples/s]40451 examples [00:43, 924.51 examples/s]40544 examples [00:43, 924.98 examples/s]40637 examples [00:44, 916.97 examples/s]40729 examples [00:44, 915.41 examples/s]40821 examples [00:44, 910.66 examples/s]40915 examples [00:44, 918.92 examples/s]41008 examples [00:44, 919.44 examples/s]41104 examples [00:44, 929.11 examples/s]41197 examples [00:44, 924.77 examples/s]41291 examples [00:44, 926.83 examples/s]41384 examples [00:44, 925.37 examples/s]41480 examples [00:44, 933.49 examples/s]41577 examples [00:45, 943.83 examples/s]41672 examples [00:45, 942.20 examples/s]41767 examples [00:45, 933.46 examples/s]41862 examples [00:45, 937.70 examples/s]41958 examples [00:45, 942.40 examples/s]42053 examples [00:45, 935.08 examples/s]42147 examples [00:45, 936.17 examples/s]42241 examples [00:45, 935.12 examples/s]42336 examples [00:45, 938.49 examples/s]42431 examples [00:45, 941.80 examples/s]42526 examples [00:46, 938.36 examples/s]42620 examples [00:46, 923.29 examples/s]42716 examples [00:46, 931.61 examples/s]42810 examples [00:46, 930.10 examples/s]42904 examples [00:46, 921.80 examples/s]42997 examples [00:46, 907.88 examples/s]43093 examples [00:46, 922.52 examples/s]43186 examples [00:46, 920.16 examples/s]43281 examples [00:46, 927.76 examples/s]43378 examples [00:46, 939.37 examples/s]43473 examples [00:47, 940.51 examples/s]43568 examples [00:47, 943.04 examples/s]43663 examples [00:47, 943.57 examples/s]43759 examples [00:47, 947.02 examples/s]43854 examples [00:47, 939.63 examples/s]43948 examples [00:47, 930.87 examples/s]44042 examples [00:47, 932.00 examples/s]44138 examples [00:47, 938.74 examples/s]44232 examples [00:47, 933.33 examples/s]44326 examples [00:47, 935.29 examples/s]44420 examples [00:48, 917.09 examples/s]44514 examples [00:48, 922.06 examples/s]44607 examples [00:48, 887.16 examples/s]44700 examples [00:48, 897.58 examples/s]44796 examples [00:48, 914.33 examples/s]44888 examples [00:48, 872.21 examples/s]44976 examples [00:48, 857.89 examples/s]45063 examples [00:48, 853.06 examples/s]45154 examples [00:48, 866.85 examples/s]45250 examples [00:49, 891.10 examples/s]45340 examples [00:49, 892.30 examples/s]45432 examples [00:49, 899.32 examples/s]45525 examples [00:49, 907.30 examples/s]45616 examples [00:49, 902.59 examples/s]45707 examples [00:49, 903.27 examples/s]45803 examples [00:49, 918.25 examples/s]45895 examples [00:49, 893.12 examples/s]45990 examples [00:49, 907.58 examples/s]46087 examples [00:49, 924.58 examples/s]46183 examples [00:50, 932.62 examples/s]46278 examples [00:50, 937.34 examples/s]46375 examples [00:50, 945.98 examples/s]46470 examples [00:50, 946.80 examples/s]46565 examples [00:50, 931.97 examples/s]46660 examples [00:50, 936.84 examples/s]46758 examples [00:50, 947.23 examples/s]46854 examples [00:50, 948.06 examples/s]46949 examples [00:50, 944.43 examples/s]47044 examples [00:50, 941.20 examples/s]47140 examples [00:51, 946.60 examples/s]47235 examples [00:51, 938.69 examples/s]47329 examples [00:51, 928.07 examples/s]47422 examples [00:51, 924.02 examples/s]47518 examples [00:51, 932.34 examples/s]47614 examples [00:51, 939.09 examples/s]47708 examples [00:51, 918.32 examples/s]47800 examples [00:51, 912.66 examples/s]47894 examples [00:51, 919.70 examples/s]47992 examples [00:51, 935.81 examples/s]48090 examples [00:52, 948.45 examples/s]48185 examples [00:52, 940.12 examples/s]48280 examples [00:52, 940.03 examples/s]48375 examples [00:52, 940.23 examples/s]48470 examples [00:52, 933.93 examples/s]48564 examples [00:52, 934.08 examples/s]48658 examples [00:52, 917.29 examples/s]48754 examples [00:52, 928.20 examples/s]48847 examples [00:52, 920.71 examples/s]48940 examples [00:53, 910.36 examples/s]49032 examples [00:53, 900.82 examples/s]49124 examples [00:53, 905.52 examples/s]49221 examples [00:53, 922.56 examples/s]49317 examples [00:53, 931.39 examples/s]49411 examples [00:53, 932.06 examples/s]49506 examples [00:53, 936.13 examples/s]49601 examples [00:53, 939.95 examples/s]49696 examples [00:53, 936.83 examples/s]49790 examples [00:53, 927.68 examples/s]49883 examples [00:54, 920.46 examples/s]49976 examples [00:54, 908.98 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▎        | 6768/50000 [00:00<00:00, 67679.58 examples/s] 37%|███▋      | 18281/50000 [00:00<00:00, 77227.57 examples/s] 60%|█████▉    | 29786/50000 [00:00<00:00, 85676.78 examples/s] 83%|████████▎ | 41309/50000 [00:00<00:00, 92816.19 examples/s]                                                               0 examples [00:00, ? examples/s]73 examples [00:00, 724.03 examples/s]167 examples [00:00, 776.82 examples/s]262 examples [00:00, 820.49 examples/s]358 examples [00:00, 856.64 examples/s]448 examples [00:00, 868.41 examples/s]542 examples [00:00, 886.50 examples/s]639 examples [00:00, 909.56 examples/s]735 examples [00:00, 922.68 examples/s]827 examples [00:00, 920.44 examples/s]919 examples [00:01, 917.16 examples/s]1011 examples [00:01, 915.60 examples/s]1102 examples [00:01, 906.41 examples/s]1192 examples [00:01, 901.78 examples/s]1290 examples [00:01, 923.10 examples/s]1383 examples [00:01, 918.41 examples/s]1475 examples [00:01, 886.31 examples/s]1564 examples [00:01, 879.64 examples/s]1658 examples [00:01, 895.97 examples/s]1757 examples [00:01, 922.07 examples/s]1850 examples [00:02, 921.14 examples/s]1946 examples [00:02, 931.66 examples/s]2041 examples [00:02, 936.21 examples/s]2140 examples [00:02, 949.67 examples/s]2236 examples [00:02, 937.11 examples/s]2332 examples [00:02, 942.83 examples/s]2431 examples [00:02, 954.66 examples/s]2528 examples [00:02, 956.50 examples/s]2624 examples [00:02, 951.55 examples/s]2720 examples [00:02, 950.86 examples/s]2816 examples [00:03, 938.85 examples/s]2912 examples [00:03, 944.01 examples/s]3007 examples [00:03, 944.67 examples/s]3104 examples [00:03, 951.18 examples/s]3200 examples [00:03, 953.56 examples/s]3296 examples [00:03, 934.13 examples/s]3392 examples [00:03, 939.31 examples/s]3492 examples [00:03, 956.33 examples/s]3590 examples [00:03, 962.29 examples/s]3689 examples [00:03, 968.38 examples/s]3786 examples [00:04, 967.70 examples/s]3883 examples [00:04, 962.18 examples/s]3980 examples [00:04, 948.26 examples/s]4075 examples [00:04, 948.38 examples/s]4170 examples [00:04, 945.32 examples/s]4265 examples [00:04, 934.38 examples/s]4359 examples [00:04, 931.67 examples/s]4453 examples [00:04, 929.93 examples/s]4547 examples [00:04, 922.01 examples/s]4642 examples [00:04, 929.14 examples/s]4735 examples [00:05, 928.94 examples/s]4830 examples [00:05, 933.98 examples/s]4925 examples [00:05, 938.24 examples/s]5021 examples [00:05, 942.58 examples/s]5116 examples [00:05, 939.63 examples/s]5212 examples [00:05, 945.19 examples/s]5309 examples [00:05, 951.02 examples/s]5407 examples [00:05, 958.01 examples/s]5503 examples [00:05, 938.56 examples/s]5599 examples [00:05, 943.63 examples/s]5694 examples [00:06, 925.79 examples/s]5787 examples [00:06, 918.52 examples/s]5879 examples [00:06, 918.27 examples/s]5977 examples [00:06, 934.25 examples/s]6071 examples [00:06, 907.99 examples/s]6167 examples [00:06, 920.98 examples/s]6261 examples [00:06, 926.47 examples/s]6356 examples [00:06, 931.35 examples/s]6451 examples [00:06, 936.67 examples/s]6547 examples [00:07, 943.02 examples/s]6642 examples [00:07, 941.55 examples/s]6739 examples [00:07, 947.86 examples/s]6834 examples [00:07, 936.78 examples/s]6928 examples [00:07, 934.67 examples/s]7022 examples [00:07, 924.16 examples/s]7116 examples [00:07, 927.33 examples/s]7213 examples [00:07, 939.31 examples/s]7307 examples [00:07, 926.01 examples/s]7401 examples [00:07, 930.16 examples/s]7499 examples [00:08, 942.76 examples/s]7596 examples [00:08, 949.42 examples/s]7692 examples [00:08, 946.93 examples/s]7789 examples [00:08, 952.64 examples/s]7885 examples [00:08, 945.20 examples/s]7980 examples [00:08, 942.97 examples/s]8075 examples [00:08, 941.08 examples/s]8170 examples [00:08, 941.13 examples/s]8265 examples [00:08, 943.38 examples/s]8360 examples [00:08, 941.40 examples/s]8455 examples [00:09, 925.04 examples/s]8548 examples [00:09, 890.87 examples/s]8642 examples [00:09, 903.63 examples/s]8738 examples [00:09, 919.46 examples/s]8831 examples [00:09, 907.94 examples/s]8926 examples [00:09, 919.55 examples/s]9022 examples [00:09, 928.87 examples/s]9120 examples [00:09, 942.25 examples/s]9217 examples [00:09, 949.37 examples/s]9315 examples [00:09, 957.30 examples/s]9413 examples [00:10, 963.71 examples/s]9510 examples [00:10, 956.33 examples/s]9607 examples [00:10, 960.04 examples/s]9704 examples [00:10, 955.82 examples/s]9801 examples [00:10, 959.71 examples/s]9898 examples [00:10, 941.72 examples/s]9994 examples [00:10, 946.70 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteMNEUM6/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteMNEUM6/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feaf1ca00d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feaf1ca00d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feaf1ca00d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feb3d67e828>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7fea7ccfa208>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7feaf1ca00d0> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7feaf1ca00d0> 

  function with postional parmater data_info <function get_dataset_torch at 0x7feaf1ca00d0> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feaefe51668>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7feaefe51780>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 14.03 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 27.42 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.18 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.18 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.21 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.21 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.60 file/s]2020-08-19 00:08:41.219022: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-08-19 00:08:41.223488: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2394455000 Hz
2020-08-19 00:08:41.223664: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x565129339c60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-19 00:08:41.223684: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  1%|          | 98304/9912422 [00:00<00:10, 972695.48it/s] 77%|███████▋  | 7585792/9912422 [00:00<00:01, 1381833.44it/s]9920512it [00:00, 43822365.88it/s]                            
0it [00:00, ?it/s]32768it [00:00, 401732.03it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 488588.45it/s]1654784it [00:00, 11585639.78it/s]                         
0it [00:00, ?it/s]8192it [00:00, 139561.97it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

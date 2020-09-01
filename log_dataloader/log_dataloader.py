
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/dev/', 'repo': 'arita37/mlmodels', 'branch': 'dev', 'sha': '8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/dev/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/8e0abaa38f42d1ec2874d5f78567a3e0c3ae98c7

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7ff4861ea6a8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7ff4861ea6a8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7ff4f0f13488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7ff4f0f13488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7ff510132ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7ff510132ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff49e249488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff49e249488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff49e249488> , (data_info, **args) 

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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 685, in parser_f
    return _read(filepath_or_buffer, kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 457, in _read
    parser = TextFileReader(fp_or_buf, **kwds)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 895, in __init__
    self._make_engine(self.engine)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1135, in _make_engine
    self._engine = CParserWrapper(self.f, **self.options)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/pandas/io/parsers.py", line 1917, in __init__
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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 416, in load
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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 25%|██▌       | 2523136/9912422 [00:00<00:00, 25001002.65it/s]9920512it [00:00, 30980463.13it/s]                             
0it [00:00, ?it/s]32768it [00:00, 597867.40it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 148959.16it/s]1654784it [00:00, 10666233.02it/s]                         
0it [00:00, ?it/s]8192it [00:00, 88616.78it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff48611df28>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff47cc65b38>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7ff49e249158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7ff49e249158> 

  function with postional parmater data_info <function tf_dataset_download at 0x7ff49e249158> , (data_info, **args) 

  CIFAR10 

  Dataset Name is :  cifar10 

Dl Completed...: 0 url [00:00, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...: 0 MiB [00:00, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/urllib3/connectionpool.py:988: InsecureRequestWarning: Unverified HTTPS request is being made to host 'www.cs.toronto.edu'. Adding certificate verification is strongly advised. See: https://urllib3.readthedocs.io/en/latest/advanced-usage.html#ssl-warnings
  InsecureRequestWarning,
Dl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   0%|          | 0/162 [00:00<?, ? MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:35,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:34,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:33,  1.69 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   3%|▎         | 5/162 [00:00<01:06,  2.37 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:06,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:05,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:05,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<01:05,  2.37 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:46,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:45,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:45,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:45,  3.29 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:31,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.55 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:31,  4.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:01<00:22,  6.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:22,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:22,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:22,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:22,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:21,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:21,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:21,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:21,  6.30 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  17%|█▋        | 28/162 [00:01<00:15,  8.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:15,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:15,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:15,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:15,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:14,  8.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:10, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:09, 11.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.96 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:07, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:06, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:06, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:06, 15.96 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 21.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 27.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 34.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.17 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 42.17 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 49.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 56.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 56.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 56.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 56.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 56.67 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 63.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 63.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 68.75 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 68.75 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 73.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 73.92 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 77.88 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 77.88 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 82.29 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 82.29 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 85.68 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.66s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.66s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.66s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 85.68 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.66s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 85.68 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 33.19 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.88s/ url]
0 examples [00:00, ? examples/s]2020-09-01 18:07:37.216725: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-01 18:07:37.233432: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-09-01 18:07:37.233676: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55918191af00 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-01 18:07:37.233696: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
25 examples [00:00, 249.04 examples/s]133 examples [00:00, 323.64 examples/s]226 examples [00:00, 402.17 examples/s]333 examples [00:00, 494.55 examples/s]437 examples [00:00, 586.89 examples/s]546 examples [00:00, 680.02 examples/s]655 examples [00:00, 765.12 examples/s]763 examples [00:00, 837.55 examples/s]863 examples [00:00, 878.69 examples/s]971 examples [00:01, 928.94 examples/s]1074 examples [00:01, 956.86 examples/s]1176 examples [00:01, 969.71 examples/s]1282 examples [00:01, 993.63 examples/s]1385 examples [00:01, 1002.81 examples/s]1493 examples [00:01, 1022.92 examples/s]1597 examples [00:01, 1018.21 examples/s]1700 examples [00:01, 1007.02 examples/s]1802 examples [00:01, 1001.34 examples/s]1907 examples [00:01, 1014.31 examples/s]2009 examples [00:02, 1006.86 examples/s]2111 examples [00:02, 995.01 examples/s] 2214 examples [00:02, 1003.02 examples/s]2315 examples [00:02, 967.93 examples/s] 2417 examples [00:02, 980.31 examples/s]2523 examples [00:02, 1002.06 examples/s]2632 examples [00:02, 1025.85 examples/s]2739 examples [00:02, 1038.23 examples/s]2845 examples [00:02, 1044.62 examples/s]2952 examples [00:02, 1051.89 examples/s]3058 examples [00:03, 1052.28 examples/s]3166 examples [00:03, 1057.90 examples/s]3275 examples [00:03, 1066.66 examples/s]3384 examples [00:03, 1071.11 examples/s]3492 examples [00:03, 1058.44 examples/s]3600 examples [00:03, 1062.89 examples/s]3709 examples [00:03, 1069.87 examples/s]3818 examples [00:03, 1075.01 examples/s]3926 examples [00:03, 1073.75 examples/s]4034 examples [00:03, 1053.34 examples/s]4140 examples [00:04, 1053.97 examples/s]4246 examples [00:04, 1049.74 examples/s]4356 examples [00:04, 1062.78 examples/s]4463 examples [00:04, 1063.78 examples/s]4571 examples [00:04, 1068.33 examples/s]4680 examples [00:04, 1074.58 examples/s]4788 examples [00:04, 1062.45 examples/s]4895 examples [00:04, 1042.21 examples/s]5002 examples [00:04, 1048.08 examples/s]5107 examples [00:04, 1028.37 examples/s]5214 examples [00:05, 1040.00 examples/s]5319 examples [00:05, 1040.25 examples/s]5428 examples [00:05, 1054.69 examples/s]5538 examples [00:05, 1065.27 examples/s]5645 examples [00:05, 1065.37 examples/s]5755 examples [00:05, 1075.01 examples/s]5863 examples [00:05, 1075.17 examples/s]5971 examples [00:05, 1055.39 examples/s]6080 examples [00:05, 1062.82 examples/s]6188 examples [00:05, 1067.59 examples/s]6298 examples [00:06, 1075.43 examples/s]6406 examples [00:06, 1075.26 examples/s]6516 examples [00:06, 1081.10 examples/s]6626 examples [00:06, 1085.45 examples/s]6735 examples [00:06, 1085.30 examples/s]6846 examples [00:06, 1091.39 examples/s]6956 examples [00:06, 1090.47 examples/s]7067 examples [00:06, 1095.35 examples/s]7177 examples [00:06, 1086.43 examples/s]7286 examples [00:07, 1074.80 examples/s]7394 examples [00:07, 1072.35 examples/s]7502 examples [00:07, 1073.67 examples/s]7610 examples [00:07, 1067.38 examples/s]7717 examples [00:07, 1054.42 examples/s]7823 examples [00:07, 1054.28 examples/s]7929 examples [00:07, 1042.23 examples/s]8034 examples [00:07, 1042.73 examples/s]8139 examples [00:07, 1041.84 examples/s]8244 examples [00:07, 1028.80 examples/s]8347 examples [00:08, 1015.52 examples/s]8449 examples [00:08, 1014.45 examples/s]8554 examples [00:08, 1023.30 examples/s]8658 examples [00:08, 1025.64 examples/s]8765 examples [00:08, 1038.00 examples/s]8870 examples [00:08, 1039.92 examples/s]8978 examples [00:08, 1049.99 examples/s]9084 examples [00:08, 1033.06 examples/s]9194 examples [00:08, 1051.36 examples/s]9304 examples [00:08, 1064.87 examples/s]9413 examples [00:09, 1069.82 examples/s]9521 examples [00:09, 1064.99 examples/s]9628 examples [00:09, 1058.43 examples/s]9734 examples [00:09, 1055.72 examples/s]9841 examples [00:09, 1057.79 examples/s]9948 examples [00:09, 1059.80 examples/s]10055 examples [00:09, 1004.23 examples/s]10157 examples [00:09, 1003.29 examples/s]10260 examples [00:09, 1010.34 examples/s]10365 examples [00:09, 1020.68 examples/s]10471 examples [00:10, 1032.07 examples/s]10575 examples [00:10, 1033.98 examples/s]10681 examples [00:10, 1039.66 examples/s]10788 examples [00:10, 1047.48 examples/s]10898 examples [00:10, 1060.69 examples/s]11005 examples [00:10, 1062.18 examples/s]11114 examples [00:10, 1068.44 examples/s]11222 examples [00:10, 1070.11 examples/s]11331 examples [00:10, 1074.34 examples/s]11442 examples [00:10, 1084.19 examples/s]11552 examples [00:11, 1086.43 examples/s]11662 examples [00:11, 1090.39 examples/s]11773 examples [00:11, 1095.87 examples/s]11883 examples [00:11, 1095.79 examples/s]11993 examples [00:11, 1095.38 examples/s]12103 examples [00:11, 1093.95 examples/s]12213 examples [00:11, 1093.60 examples/s]12323 examples [00:11, 1086.55 examples/s]12432 examples [00:11, 1082.97 examples/s]12542 examples [00:11, 1086.90 examples/s]12651 examples [00:12, 1087.45 examples/s]12761 examples [00:12, 1090.91 examples/s]12871 examples [00:12, 1089.79 examples/s]12980 examples [00:12, 1088.58 examples/s]13089 examples [00:12, 1084.75 examples/s]13198 examples [00:12, 1084.58 examples/s]13307 examples [00:12, 1085.42 examples/s]13416 examples [00:12, 1072.74 examples/s]13524 examples [00:12, 1074.31 examples/s]13634 examples [00:12, 1080.93 examples/s]13743 examples [00:13, 1083.12 examples/s]13852 examples [00:13, 1083.87 examples/s]13961 examples [00:13, 1078.73 examples/s]14069 examples [00:13, 1073.89 examples/s]14178 examples [00:13, 1078.13 examples/s]14287 examples [00:13, 1080.66 examples/s]14396 examples [00:13, 1059.80 examples/s]14504 examples [00:13, 1065.19 examples/s]14611 examples [00:13, 1049.98 examples/s]14722 examples [00:14, 1065.10 examples/s]14831 examples [00:14, 1070.47 examples/s]14941 examples [00:14, 1077.50 examples/s]15052 examples [00:14, 1084.53 examples/s]15161 examples [00:14, 1071.04 examples/s]15269 examples [00:14, 1054.61 examples/s]15375 examples [00:14, 1050.55 examples/s]15481 examples [00:14, 1040.04 examples/s]15588 examples [00:14, 1048.70 examples/s]15693 examples [00:14, 1046.44 examples/s]15802 examples [00:15, 1058.88 examples/s]15908 examples [00:15, 1044.97 examples/s]16017 examples [00:15, 1055.69 examples/s]16126 examples [00:15, 1065.48 examples/s]16235 examples [00:15, 1069.96 examples/s]16343 examples [00:15, 1071.43 examples/s]16451 examples [00:15, 1070.68 examples/s]16559 examples [00:15, 1072.15 examples/s]16667 examples [00:15, 1069.99 examples/s]16775 examples [00:15, 1054.97 examples/s]16881 examples [00:16, 1055.30 examples/s]16989 examples [00:16, 1060.34 examples/s]17096 examples [00:16, 1045.16 examples/s]17201 examples [00:16, 1033.67 examples/s]17306 examples [00:16, 1036.29 examples/s]17410 examples [00:16, 1026.83 examples/s]17515 examples [00:16, 1031.16 examples/s]17623 examples [00:16, 1044.04 examples/s]17731 examples [00:16, 1052.00 examples/s]17837 examples [00:16, 1053.25 examples/s]17943 examples [00:17, 1050.94 examples/s]18049 examples [00:17, 1012.31 examples/s]18157 examples [00:17, 1029.24 examples/s]18265 examples [00:17, 1041.81 examples/s]18370 examples [00:17, 1040.45 examples/s]18475 examples [00:17, 1034.32 examples/s]18579 examples [00:17, 1006.97 examples/s]18687 examples [00:17, 1027.52 examples/s]18791 examples [00:17, 1029.29 examples/s]18899 examples [00:17, 1043.90 examples/s]19007 examples [00:18, 1051.88 examples/s]19114 examples [00:18, 1054.52 examples/s]19224 examples [00:18, 1064.93 examples/s]19332 examples [00:18, 1069.16 examples/s]19439 examples [00:18, 1054.13 examples/s]19546 examples [00:18, 1057.71 examples/s]19652 examples [00:18, 1010.39 examples/s]19754 examples [00:18, 994.61 examples/s] 19863 examples [00:18, 1019.09 examples/s]19972 examples [00:19, 1038.22 examples/s]20077 examples [00:19, 992.74 examples/s] 20186 examples [00:19, 1018.43 examples/s]20289 examples [00:19, 985.93 examples/s] 20399 examples [00:19, 1015.82 examples/s]20509 examples [00:19, 1037.77 examples/s]20618 examples [00:19, 1051.13 examples/s]20724 examples [00:19, 1040.22 examples/s]20833 examples [00:19, 1054.27 examples/s]20940 examples [00:19, 1058.10 examples/s]21048 examples [00:20, 1063.10 examples/s]21155 examples [00:20, 1060.35 examples/s]21262 examples [00:20, 1063.20 examples/s]21369 examples [00:20, 1026.31 examples/s]21480 examples [00:20, 1047.73 examples/s]21591 examples [00:20, 1064.13 examples/s]21698 examples [00:20, 1064.25 examples/s]21807 examples [00:20, 1070.18 examples/s]21917 examples [00:20, 1078.39 examples/s]22025 examples [00:20, 1072.13 examples/s]22133 examples [00:21, 1063.10 examples/s]22240 examples [00:21, 1039.35 examples/s]22345 examples [00:21, 1019.85 examples/s]22451 examples [00:21, 1030.31 examples/s]22557 examples [00:21, 1038.93 examples/s]22662 examples [00:21, 1028.90 examples/s]22766 examples [00:21, 1031.61 examples/s]22871 examples [00:21, 1035.37 examples/s]22975 examples [00:21, 1024.11 examples/s]23078 examples [00:22, 1019.50 examples/s]23181 examples [00:22, 1009.71 examples/s]23283 examples [00:22, 999.94 examples/s] 23385 examples [00:22, 1005.68 examples/s]23493 examples [00:22, 1024.41 examples/s]23598 examples [00:22, 1030.58 examples/s]23703 examples [00:22, 1035.90 examples/s]23810 examples [00:22, 1045.02 examples/s]23918 examples [00:22, 1052.80 examples/s]24024 examples [00:22, 1051.64 examples/s]24131 examples [00:23, 1056.78 examples/s]24240 examples [00:23, 1065.32 examples/s]24347 examples [00:23, 1060.09 examples/s]24454 examples [00:23, 1050.35 examples/s]24560 examples [00:23, 1051.82 examples/s]24666 examples [00:23, 1011.81 examples/s]24772 examples [00:23, 1025.27 examples/s]24882 examples [00:23, 1042.18 examples/s]24992 examples [00:23, 1056.65 examples/s]25101 examples [00:23, 1063.82 examples/s]25210 examples [00:24, 1069.28 examples/s]25319 examples [00:24, 1073.22 examples/s]25428 examples [00:24, 1075.85 examples/s]25536 examples [00:24, 1075.93 examples/s]25647 examples [00:24, 1083.94 examples/s]25758 examples [00:24, 1090.05 examples/s]25868 examples [00:24, 1081.52 examples/s]25977 examples [00:24, 1048.89 examples/s]26084 examples [00:24, 1052.80 examples/s]26190 examples [00:24, 1042.55 examples/s]26295 examples [00:25, 1038.25 examples/s]26399 examples [00:25, 975.07 examples/s] 26501 examples [00:25, 987.71 examples/s]26607 examples [00:25, 1005.91 examples/s]26709 examples [00:25, 984.91 examples/s] 26818 examples [00:25, 1013.89 examples/s]26922 examples [00:25, 1021.44 examples/s]27025 examples [00:25, 1017.02 examples/s]27131 examples [00:25, 1028.95 examples/s]27236 examples [00:26, 1032.42 examples/s]27340 examples [00:26, 997.58 examples/s] 27443 examples [00:26, 1006.18 examples/s]27549 examples [00:26, 1019.21 examples/s]27652 examples [00:26, 1001.64 examples/s]27756 examples [00:26, 1010.97 examples/s]27858 examples [00:26, 1012.66 examples/s]27965 examples [00:26, 1026.73 examples/s]28068 examples [00:26, 1010.02 examples/s]28170 examples [00:26, 988.79 examples/s] 28272 examples [00:27, 994.10 examples/s]28372 examples [00:27, 965.05 examples/s]28471 examples [00:27, 971.37 examples/s]28574 examples [00:27, 985.35 examples/s]28673 examples [00:27, 977.52 examples/s]28776 examples [00:27, 992.33 examples/s]28878 examples [00:27, 998.61 examples/s]28978 examples [00:27, 998.47 examples/s]29083 examples [00:27, 1011.59 examples/s]29185 examples [00:27, 982.71 examples/s] 29293 examples [00:28, 1007.18 examples/s]29395 examples [00:28, 1001.08 examples/s]29500 examples [00:28, 1012.71 examples/s]29602 examples [00:28, 1006.52 examples/s]29707 examples [00:28, 1018.11 examples/s]29812 examples [00:28, 1024.78 examples/s]29919 examples [00:28, 1035.64 examples/s]30023 examples [00:28, 997.92 examples/s] 30131 examples [00:28, 1019.83 examples/s]30237 examples [00:28, 1028.96 examples/s]30345 examples [00:29, 1041.54 examples/s]30452 examples [00:29, 1049.63 examples/s]30558 examples [00:29, 1038.50 examples/s]30663 examples [00:29, 1002.73 examples/s]30764 examples [00:29, 998.22 examples/s] 30869 examples [00:29, 1010.47 examples/s]30979 examples [00:29, 1033.67 examples/s]31088 examples [00:29, 1048.47 examples/s]31194 examples [00:29, 1039.62 examples/s]31299 examples [00:30, 1034.81 examples/s]31403 examples [00:30, 1019.89 examples/s]31506 examples [00:30, 1016.07 examples/s]31608 examples [00:30, 1001.25 examples/s]31709 examples [00:30, 991.31 examples/s] 31814 examples [00:30, 1006.73 examples/s]31915 examples [00:30, 996.70 examples/s] 32015 examples [00:30, 978.38 examples/s]32122 examples [00:30, 1003.52 examples/s]32228 examples [00:30, 1019.66 examples/s]32336 examples [00:31, 1034.69 examples/s]32443 examples [00:31, 1042.40 examples/s]32548 examples [00:31, 1020.00 examples/s]32658 examples [00:31, 1041.82 examples/s]32764 examples [00:31, 1046.43 examples/s]32869 examples [00:31, 1045.22 examples/s]32974 examples [00:31, 1040.51 examples/s]33079 examples [00:31, 1004.09 examples/s]33180 examples [00:31, 982.64 examples/s] 33280 examples [00:31, 986.04 examples/s]33379 examples [00:32, 980.61 examples/s]33478 examples [00:32, 976.36 examples/s]33576 examples [00:32, 959.35 examples/s]33679 examples [00:32, 978.37 examples/s]33784 examples [00:32, 998.73 examples/s]33889 examples [00:32, 1011.56 examples/s]33994 examples [00:32, 1020.22 examples/s]34097 examples [00:32, 987.11 examples/s] 34197 examples [00:32, 989.45 examples/s]34301 examples [00:33, 1001.54 examples/s]34404 examples [00:33, 1009.36 examples/s]34506 examples [00:33, 1011.71 examples/s]34608 examples [00:33, 1012.90 examples/s]34714 examples [00:33, 1025.58 examples/s]34817 examples [00:33, 1021.37 examples/s]34920 examples [00:33, 1002.57 examples/s]35023 examples [00:33, 1010.24 examples/s]35125 examples [00:33, 982.64 examples/s] 35224 examples [00:33, 973.58 examples/s]35322 examples [00:34, 960.23 examples/s]35426 examples [00:34, 980.45 examples/s]35530 examples [00:34, 994.23 examples/s]35637 examples [00:34, 1015.15 examples/s]35739 examples [00:34, 1006.55 examples/s]35840 examples [00:34, 980.96 examples/s] 35947 examples [00:34, 1005.12 examples/s]36057 examples [00:34, 1029.24 examples/s]36161 examples [00:34, 999.56 examples/s] 36268 examples [00:34, 1018.50 examples/s]36376 examples [00:35, 1035.21 examples/s]36480 examples [00:35, 1033.96 examples/s]36591 examples [00:35, 1053.05 examples/s]36697 examples [00:35, 1012.93 examples/s]36803 examples [00:35, 1025.84 examples/s]36910 examples [00:35, 1038.39 examples/s]37015 examples [00:35, 1041.04 examples/s]37122 examples [00:35, 1047.21 examples/s]37227 examples [00:35, 1029.95 examples/s]37331 examples [00:36, 983.40 examples/s] 37439 examples [00:36, 1009.73 examples/s]37541 examples [00:36, 987.00 examples/s] 37641 examples [00:36, 972.64 examples/s]37744 examples [00:36, 981.82 examples/s]37843 examples [00:36, 910.51 examples/s]37948 examples [00:36, 946.41 examples/s]38051 examples [00:36, 970.01 examples/s]38157 examples [00:36, 993.89 examples/s]38258 examples [00:36, 986.96 examples/s]38358 examples [00:37, 990.42 examples/s]38458 examples [00:37, 992.78 examples/s]38562 examples [00:37, 1005.56 examples/s]38665 examples [00:37, 1011.06 examples/s]38769 examples [00:37, 1017.63 examples/s]38878 examples [00:37, 1037.85 examples/s]38982 examples [00:37, 1033.54 examples/s]39088 examples [00:37, 1041.05 examples/s]39193 examples [00:37, 1030.51 examples/s]39299 examples [00:37, 1036.82 examples/s]39403 examples [00:38, 1009.39 examples/s]39505 examples [00:38, 997.61 examples/s] 39611 examples [00:38, 1013.27 examples/s]39719 examples [00:38, 1031.47 examples/s]39823 examples [00:38, 1032.95 examples/s]39927 examples [00:38, 1012.31 examples/s]40029 examples [00:38, 950.14 examples/s] 40135 examples [00:38, 979.85 examples/s]40242 examples [00:38, 1003.06 examples/s]40349 examples [00:39, 1020.63 examples/s]40453 examples [00:39, 1023.60 examples/s]40559 examples [00:39, 1033.69 examples/s]40669 examples [00:39, 1050.33 examples/s]40775 examples [00:39, 1023.86 examples/s]40878 examples [00:39, 1025.69 examples/s]40983 examples [00:39, 1032.49 examples/s]41090 examples [00:39, 1040.75 examples/s]41199 examples [00:39, 1054.42 examples/s]41307 examples [00:39, 1059.34 examples/s]41414 examples [00:40, 1060.64 examples/s]41521 examples [00:40, 1052.11 examples/s]41629 examples [00:40, 1059.93 examples/s]41740 examples [00:40, 1073.84 examples/s]41851 examples [00:40, 1083.11 examples/s]41960 examples [00:40, 1082.13 examples/s]42069 examples [00:40, 1075.20 examples/s]42177 examples [00:40, 1076.43 examples/s]42285 examples [00:40, 1071.13 examples/s]42393 examples [00:40, 1065.15 examples/s]42500 examples [00:41, 1040.38 examples/s]42605 examples [00:41, 1001.78 examples/s]42713 examples [00:41, 1023.23 examples/s]42820 examples [00:41, 1034.36 examples/s]42926 examples [00:41, 1039.61 examples/s]43035 examples [00:41, 1051.77 examples/s]43144 examples [00:41, 1060.32 examples/s]43251 examples [00:41, 1060.00 examples/s]43358 examples [00:41, 1043.19 examples/s]43464 examples [00:41, 1047.90 examples/s]43569 examples [00:42, 1045.76 examples/s]43676 examples [00:42, 1051.18 examples/s]43782 examples [00:42, 1035.72 examples/s]43886 examples [00:42, 1000.80 examples/s]43989 examples [00:42, 1008.13 examples/s]44092 examples [00:42, 1014.37 examples/s]44194 examples [00:42, 1014.89 examples/s]44298 examples [00:42, 1020.06 examples/s]44406 examples [00:42, 1035.21 examples/s]44516 examples [00:43, 1051.57 examples/s]44624 examples [00:43, 1059.83 examples/s]44732 examples [00:43, 1063.87 examples/s]44839 examples [00:43, 1053.98 examples/s]44945 examples [00:43, 976.75 examples/s] 45046 examples [00:43, 985.34 examples/s]45146 examples [00:43, 970.34 examples/s]45254 examples [00:43, 1000.44 examples/s]45357 examples [00:43, 1006.87 examples/s]45459 examples [00:43, 996.43 examples/s] 45560 examples [00:44, 987.15 examples/s]45659 examples [00:44, 972.15 examples/s]45764 examples [00:44, 993.88 examples/s]45869 examples [00:44, 1007.73 examples/s]45976 examples [00:44, 1023.81 examples/s]46082 examples [00:44, 1034.01 examples/s]46188 examples [00:44, 1039.06 examples/s]46293 examples [00:44, 1040.64 examples/s]46398 examples [00:44, 1043.14 examples/s]46504 examples [00:44, 1045.51 examples/s]46609 examples [00:45, 1015.21 examples/s]46713 examples [00:45, 1022.19 examples/s]46817 examples [00:45, 1026.18 examples/s]46926 examples [00:45, 1044.49 examples/s]47033 examples [00:45, 1051.03 examples/s]47139 examples [00:45, 1042.03 examples/s]47244 examples [00:45, 1043.50 examples/s]47349 examples [00:45, 1037.84 examples/s]47455 examples [00:45, 1041.68 examples/s]47560 examples [00:45, 1022.38 examples/s]47663 examples [00:46, 1015.38 examples/s]47765 examples [00:46, 994.73 examples/s] 47865 examples [00:46, 995.92 examples/s]47966 examples [00:46, 997.77 examples/s]48073 examples [00:46, 1016.89 examples/s]48177 examples [00:46, 1020.73 examples/s]48280 examples [00:46, 1017.82 examples/s]48382 examples [00:46, 1009.05 examples/s]48483 examples [00:46, 968.67 examples/s] 48589 examples [00:47, 993.78 examples/s]48689 examples [00:47, 991.90 examples/s]48795 examples [00:47, 1010.07 examples/s]48899 examples [00:47, 1016.79 examples/s]49007 examples [00:47, 1033.95 examples/s]49113 examples [00:47, 1041.62 examples/s]49218 examples [00:47, 1005.72 examples/s]49324 examples [00:47, 1019.91 examples/s]49432 examples [00:47, 1034.81 examples/s]49536 examples [00:47, 1023.84 examples/s]49639 examples [00:48, 1016.21 examples/s]49741 examples [00:48, 1015.83 examples/s]49843 examples [00:48, 996.24 examples/s] 49946 examples [00:48, 1004.58 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 11%|█         | 5579/50000 [00:00<00:00, 55788.32 examples/s] 37%|███▋      | 18297/50000 [00:00<00:00, 67084.95 examples/s] 62%|██████▏   | 31210/50000 [00:00<00:00, 78383.44 examples/s] 89%|████████▊ | 44296/50000 [00:00<00:00, 89101.53 examples/s]                                                               0 examples [00:00, ? examples/s]87 examples [00:00, 869.69 examples/s]186 examples [00:00, 901.94 examples/s]293 examples [00:00, 945.41 examples/s]398 examples [00:00, 973.96 examples/s]508 examples [00:00, 1006.52 examples/s]609 examples [00:00, 1007.36 examples/s]719 examples [00:00, 1031.43 examples/s]827 examples [00:00, 1045.49 examples/s]937 examples [00:00, 1060.96 examples/s]1040 examples [00:01, 1031.86 examples/s]1143 examples [00:01, 1029.91 examples/s]1251 examples [00:01, 1040.26 examples/s]1357 examples [00:01, 1045.04 examples/s]1461 examples [00:01, 1030.76 examples/s]1569 examples [00:01, 1044.64 examples/s]1677 examples [00:01, 1051.86 examples/s]1786 examples [00:01, 1061.18 examples/s]1893 examples [00:01, 1051.07 examples/s]1999 examples [00:01, 1053.31 examples/s]2108 examples [00:02, 1061.49 examples/s]2215 examples [00:02, 1053.12 examples/s]2321 examples [00:02, 1045.95 examples/s]2426 examples [00:02, 1008.25 examples/s]2535 examples [00:02, 1029.91 examples/s]2643 examples [00:02, 1043.41 examples/s]2748 examples [00:02, 1035.66 examples/s]2855 examples [00:02, 1043.79 examples/s]2960 examples [00:02, 1045.20 examples/s]3065 examples [00:02, 1022.86 examples/s]3168 examples [00:03, 1021.12 examples/s]3272 examples [00:03, 1024.96 examples/s]3377 examples [00:03, 1029.91 examples/s]3485 examples [00:03, 1044.04 examples/s]3594 examples [00:03, 1055.01 examples/s]3701 examples [00:03, 1059.34 examples/s]3808 examples [00:03, 1051.16 examples/s]3914 examples [00:03, 1050.09 examples/s]4021 examples [00:03, 1054.20 examples/s]4127 examples [00:03, 1049.57 examples/s]4232 examples [00:04, 1027.69 examples/s]4338 examples [00:04, 1036.24 examples/s]4447 examples [00:04, 1050.60 examples/s]4557 examples [00:04, 1064.16 examples/s]4666 examples [00:04, 1069.93 examples/s]4774 examples [00:04, 849.40 examples/s] 4884 examples [00:04, 910.59 examples/s]4993 examples [00:04, 955.83 examples/s]5103 examples [00:04, 994.47 examples/s]5214 examples [00:05, 1025.35 examples/s]5324 examples [00:05, 1046.15 examples/s]5433 examples [00:05, 1057.93 examples/s]5541 examples [00:05, 1058.98 examples/s]5651 examples [00:05, 1069.53 examples/s]5759 examples [00:05, 1062.60 examples/s]5869 examples [00:05, 1071.52 examples/s]5977 examples [00:05, 1037.66 examples/s]6082 examples [00:05, 1022.81 examples/s]6185 examples [00:05, 1012.36 examples/s]6291 examples [00:06, 1025.93 examples/s]6401 examples [00:06, 1046.12 examples/s]6507 examples [00:06, 1049.75 examples/s]6617 examples [00:06, 1062.99 examples/s]6726 examples [00:06, 1068.99 examples/s]6836 examples [00:06, 1076.00 examples/s]6945 examples [00:06, 1078.75 examples/s]7054 examples [00:06, 1080.17 examples/s]7163 examples [00:06, 1081.68 examples/s]7273 examples [00:06, 1086.79 examples/s]7382 examples [00:07, 1062.28 examples/s]7489 examples [00:07, 1063.85 examples/s]7596 examples [00:07, 1051.57 examples/s]7702 examples [00:07, 1041.91 examples/s]7807 examples [00:07, 1036.93 examples/s]7915 examples [00:07, 1046.87 examples/s]8023 examples [00:07, 1055.42 examples/s]8130 examples [00:07, 1057.41 examples/s]8238 examples [00:07, 1063.77 examples/s]8345 examples [00:08, 1058.01 examples/s]8451 examples [00:08, 1056.22 examples/s]8558 examples [00:08, 1057.84 examples/s]8664 examples [00:08, 1047.61 examples/s]8769 examples [00:08, 1040.55 examples/s]8874 examples [00:08, 1041.41 examples/s]8979 examples [00:08, 1043.41 examples/s]9084 examples [00:08, 1032.47 examples/s]9189 examples [00:08, 1035.32 examples/s]9294 examples [00:08, 1038.65 examples/s]9398 examples [00:09, 1024.63 examples/s]9501 examples [00:09, 1004.42 examples/s]9605 examples [00:09, 1013.93 examples/s]9707 examples [00:09, 1015.21 examples/s]9812 examples [00:09, 1025.28 examples/s]9922 examples [00:09, 1044.96 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteDCR91O/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteDCR91O/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff49e249488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff49e249488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff49e249488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff49c2a7240>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff4244b0198>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7ff49e249488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7ff49e249488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7ff49e249488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff47cc892b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7ff47cc89160>), {}) 

  




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
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/importlib/__init__.py", line 126, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 994, in _gcd_import
  File "<frozen importlib._bootstrap>", line 971, in _find_and_load
  File "<frozen importlib._bootstrap>", line 955, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 665, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 678, in exec_module
  File "<frozen importlib._bootstrap>", line 219, in _call_with_frames_removed
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/model_tch/textcnn.py", line 24, in <module>
    import torchtext
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 42, in <module>
    _init_extension()
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torchtext/__init__.py", line 38, in _init_extension
    torch.ops.load_library(ext_specs.origin)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/site-packages/torch/_ops.py", line 106, in load_library
    ctypes.CDLL(path)
  File "/opt/hostedtoolcache/Python/3.6.12/x64/lib/python3.6/ctypes/__init__.py", line 348, in __init__
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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 10.35 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.54 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 19.54 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.39 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  9.39 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.55 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.55 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  6.66 file/s]2020-09-01 18:08:43.194053: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-09-01 18:08:43.198488: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095225000 Hz
2020-09-01 18:08:43.198711: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5581cc671560 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-01 18:08:43.198734: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'mnist2', 'fashion-mnist_small.npy', 'mnist_dataset_small.npy', 'train', 'test'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 157672.43it/s] 81%|████████  | 8028160/9912422 [00:00<00:08, 225049.95it/s]9920512it [00:00, 45095625.18it/s]                           
0it [00:00, ?it/s]32768it [00:00, 576068.31it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 160501.02it/s]1654784it [00:00, 10940849.66it/s]                         
0it [00:00, ?it/s]8192it [00:00, 146435.98it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

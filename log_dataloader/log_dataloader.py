
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f2197409620> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f2197409620> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f2202132488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f2202132488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f2221351ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f2221351ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f21af468488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f21af468488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f21af468488> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 30%|██▉       | 2949120/9912422 [00:00<00:00, 29425562.04it/s]9920512it [00:00, 30909948.22it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 305552.98it/s]           
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 149185.85it/s]1654784it [00:00, 10532503.69it/s]                         
0it [00:00, ?it/s]8192it [00:00, 171535.39it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f219733eef0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f21966857b8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f21af468158> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f21af468158> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f21af468158> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:16,  2.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:54,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:51,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.86 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  4.03 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:35,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  4.03 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:23,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.80 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:16,  7.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:16,  7.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.80 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:15,  7.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:15,  7.80 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.72 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.72 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:05, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:05, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:05, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:05, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:05, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 24.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.44 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 31.44 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 38.38 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.62 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 45.62 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.92 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:00, 52.92 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 58.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 64.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 68.71 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 72.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 72.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 74.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.83 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  2.52s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 75.83 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 31.86 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.08s/ url]
0 examples [00:00, ? examples/s]2020-09-03 00:08:30.380401: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-03 00:08:30.391534: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-09-03 00:08:30.392339: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55cd3e9f05d0 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 00:08:30.392366: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
13 examples [00:00, 129.03 examples/s]97 examples [00:00, 172.92 examples/s]169 examples [00:00, 222.82 examples/s]246 examples [00:00, 282.89 examples/s]332 examples [00:00, 353.51 examples/s]418 examples [00:00, 428.73 examples/s]507 examples [00:00, 507.62 examples/s]593 examples [00:00, 578.57 examples/s]676 examples [00:00, 635.29 examples/s]764 examples [00:01, 691.76 examples/s]851 examples [00:01, 736.08 examples/s]936 examples [00:01, 766.49 examples/s]1022 examples [00:01, 790.15 examples/s]1107 examples [00:01, 798.35 examples/s]1191 examples [00:01, 803.47 examples/s]1276 examples [00:01, 814.59 examples/s]1362 examples [00:01, 826.11 examples/s]1452 examples [00:01, 844.55 examples/s]1538 examples [00:01, 840.63 examples/s]1623 examples [00:02, 824.11 examples/s]1710 examples [00:02, 835.35 examples/s]1795 examples [00:02, 837.63 examples/s]1885 examples [00:02, 854.85 examples/s]1971 examples [00:02, 856.20 examples/s]2059 examples [00:02, 863.17 examples/s]2148 examples [00:02, 868.92 examples/s]2236 examples [00:02, 869.07 examples/s]2328 examples [00:02, 881.95 examples/s]2417 examples [00:02, 869.62 examples/s]2505 examples [00:03, 853.69 examples/s]2591 examples [00:03, 849.91 examples/s]2680 examples [00:03, 861.53 examples/s]2772 examples [00:03, 875.85 examples/s]2860 examples [00:03, 806.85 examples/s]2949 examples [00:03, 827.71 examples/s]3040 examples [00:03, 848.85 examples/s]3126 examples [00:03, 840.88 examples/s]3211 examples [00:03, 818.37 examples/s]3294 examples [00:04, 810.13 examples/s]3376 examples [00:04, 804.89 examples/s]3457 examples [00:04, 803.84 examples/s]3538 examples [00:04, 801.52 examples/s]3619 examples [00:04, 795.61 examples/s]3699 examples [00:04, 796.33 examples/s]3783 examples [00:04, 807.90 examples/s]3868 examples [00:04, 820.07 examples/s]3951 examples [00:04, 814.09 examples/s]4037 examples [00:04, 826.61 examples/s]4127 examples [00:05, 845.69 examples/s]4216 examples [00:05, 856.11 examples/s]4302 examples [00:05, 856.50 examples/s]4393 examples [00:05, 870.16 examples/s]4481 examples [00:05, 855.17 examples/s]4569 examples [00:05, 861.93 examples/s]4656 examples [00:05, 863.12 examples/s]4744 examples [00:05, 867.49 examples/s]4831 examples [00:05, 854.07 examples/s]4917 examples [00:05, 850.29 examples/s]5007 examples [00:06, 861.08 examples/s]5094 examples [00:06, 853.11 examples/s]5182 examples [00:06, 860.62 examples/s]5272 examples [00:06, 868.55 examples/s]5359 examples [00:06, 827.13 examples/s]5443 examples [00:06, 827.08 examples/s]5531 examples [00:06, 841.50 examples/s]5616 examples [00:06, 832.05 examples/s]5700 examples [00:06, 810.55 examples/s]5787 examples [00:06, 825.40 examples/s]5870 examples [00:07, 801.77 examples/s]5953 examples [00:07, 808.41 examples/s]6039 examples [00:07, 821.90 examples/s]6126 examples [00:07, 833.84 examples/s]6210 examples [00:07, 808.78 examples/s]6298 examples [00:07, 827.18 examples/s]6384 examples [00:07, 834.20 examples/s]6475 examples [00:07, 853.89 examples/s]6565 examples [00:07, 866.55 examples/s]6652 examples [00:08, 841.69 examples/s]6737 examples [00:08, 838.25 examples/s]6822 examples [00:08, 831.57 examples/s]6913 examples [00:08, 851.15 examples/s]7000 examples [00:08, 856.37 examples/s]7087 examples [00:08, 858.26 examples/s]7173 examples [00:08, 847.18 examples/s]7258 examples [00:08, 791.30 examples/s]7345 examples [00:08, 811.45 examples/s]7432 examples [00:08, 826.28 examples/s]7516 examples [00:09, 818.62 examples/s]7599 examples [00:09, 803.92 examples/s]7687 examples [00:09, 823.44 examples/s]7778 examples [00:09, 846.21 examples/s]7864 examples [00:09, 832.05 examples/s]7952 examples [00:09, 843.52 examples/s]8044 examples [00:09, 864.38 examples/s]8133 examples [00:09, 869.98 examples/s]8226 examples [00:09, 887.13 examples/s]8315 examples [00:09, 855.60 examples/s]8407 examples [00:10, 871.92 examples/s]8495 examples [00:10, 874.29 examples/s]8583 examples [00:10, 875.03 examples/s]8671 examples [00:10, 843.56 examples/s]8756 examples [00:10, 840.05 examples/s]8843 examples [00:10, 846.56 examples/s]8934 examples [00:10, 862.39 examples/s]9021 examples [00:10, 834.05 examples/s]9105 examples [00:10, 806.52 examples/s]9187 examples [00:11, 788.64 examples/s]9275 examples [00:11, 812.27 examples/s]9359 examples [00:11, 820.07 examples/s]9445 examples [00:11, 831.22 examples/s]9537 examples [00:11, 853.66 examples/s]9623 examples [00:11, 851.06 examples/s]9713 examples [00:11, 864.30 examples/s]9803 examples [00:11, 872.71 examples/s]9891 examples [00:11, 863.29 examples/s]9978 examples [00:11, 829.35 examples/s]10062 examples [00:12, 754.45 examples/s]10149 examples [00:12, 783.69 examples/s]10234 examples [00:12, 801.56 examples/s]10321 examples [00:12, 820.11 examples/s]10407 examples [00:12, 830.80 examples/s]10491 examples [00:12, 813.06 examples/s]10574 examples [00:12, 816.22 examples/s]10656 examples [00:12, 814.54 examples/s]10745 examples [00:12, 834.63 examples/s]10830 examples [00:13, 836.54 examples/s]10920 examples [00:13, 851.91 examples/s]11010 examples [00:13, 864.06 examples/s]11099 examples [00:13, 870.38 examples/s]11189 examples [00:13, 878.74 examples/s]11277 examples [00:13, 867.45 examples/s]11365 examples [00:13, 869.23 examples/s]11453 examples [00:13, 834.82 examples/s]11537 examples [00:13, 823.74 examples/s]11620 examples [00:13, 801.73 examples/s]11701 examples [00:14, 787.84 examples/s]11792 examples [00:14, 819.81 examples/s]11877 examples [00:14, 827.76 examples/s]11970 examples [00:14, 854.15 examples/s]12064 examples [00:14, 876.06 examples/s]12153 examples [00:14, 875.15 examples/s]12244 examples [00:14, 884.17 examples/s]12334 examples [00:14, 888.64 examples/s]12426 examples [00:14, 895.45 examples/s]12519 examples [00:14, 904.32 examples/s]12610 examples [00:15, 904.57 examples/s]12702 examples [00:15, 907.31 examples/s]12796 examples [00:15, 916.26 examples/s]12890 examples [00:15, 923.19 examples/s]12988 examples [00:15, 938.74 examples/s]13082 examples [00:15, 902.80 examples/s]13175 examples [00:15, 908.72 examples/s]13268 examples [00:15, 913.53 examples/s]13361 examples [00:15, 916.81 examples/s]13455 examples [00:15, 923.27 examples/s]13548 examples [00:16, 916.83 examples/s]13640 examples [00:16, 911.42 examples/s]13735 examples [00:16, 920.37 examples/s]13828 examples [00:16, 886.90 examples/s]13917 examples [00:16, 885.27 examples/s]14006 examples [00:16, 837.14 examples/s]14092 examples [00:16, 841.81 examples/s]14183 examples [00:16, 858.31 examples/s]14270 examples [00:16, 858.79 examples/s]14357 examples [00:17, 857.82 examples/s]14444 examples [00:17, 860.19 examples/s]14531 examples [00:17, 821.20 examples/s]14616 examples [00:17, 827.61 examples/s]14707 examples [00:17, 844.69 examples/s]14792 examples [00:17, 846.06 examples/s]14883 examples [00:17, 862.22 examples/s]14975 examples [00:17, 877.72 examples/s]15066 examples [00:17, 886.19 examples/s]15155 examples [00:17, 873.00 examples/s]15243 examples [00:18, 840.89 examples/s]15329 examples [00:18, 845.65 examples/s]15419 examples [00:18, 860.55 examples/s]15510 examples [00:18, 873.39 examples/s]15598 examples [00:18, 861.18 examples/s]15686 examples [00:18, 865.41 examples/s]15773 examples [00:18, 834.61 examples/s]15857 examples [00:18, 832.86 examples/s]15946 examples [00:18, 847.17 examples/s]16031 examples [00:18, 839.22 examples/s]16116 examples [00:19, 828.15 examples/s]16199 examples [00:19, 826.01 examples/s]16285 examples [00:19, 835.18 examples/s]16371 examples [00:19, 841.85 examples/s]16456 examples [00:19, 834.02 examples/s]16540 examples [00:19, 822.46 examples/s]16627 examples [00:19, 835.55 examples/s]16712 examples [00:19, 839.17 examples/s]16796 examples [00:19, 806.75 examples/s]16880 examples [00:20, 813.36 examples/s]16962 examples [00:20, 798.86 examples/s]17045 examples [00:20, 807.14 examples/s]17131 examples [00:20, 821.76 examples/s]17214 examples [00:20, 814.08 examples/s]17301 examples [00:20, 830.06 examples/s]17385 examples [00:20, 817.55 examples/s]17467 examples [00:20, 800.67 examples/s]17548 examples [00:20, 799.84 examples/s]17629 examples [00:20, 802.22 examples/s]17712 examples [00:21, 810.01 examples/s]17794 examples [00:21, 811.85 examples/s]17879 examples [00:21, 821.50 examples/s]17962 examples [00:21, 815.84 examples/s]18049 examples [00:21, 830.34 examples/s]18133 examples [00:21, 823.55 examples/s]18216 examples [00:21, 818.97 examples/s]18303 examples [00:21, 833.23 examples/s]18388 examples [00:21, 836.34 examples/s]18475 examples [00:21, 845.23 examples/s]18561 examples [00:22, 848.53 examples/s]18646 examples [00:22, 826.56 examples/s]18734 examples [00:22, 839.96 examples/s]18823 examples [00:22, 851.77 examples/s]18912 examples [00:22, 862.56 examples/s]19003 examples [00:22, 875.71 examples/s]19091 examples [00:22, 871.84 examples/s]19179 examples [00:22, 872.71 examples/s]19276 examples [00:22, 899.64 examples/s]19369 examples [00:22, 907.44 examples/s]19461 examples [00:23, 909.40 examples/s]19553 examples [00:23, 910.68 examples/s]19645 examples [00:23, 891.46 examples/s]19736 examples [00:23, 896.69 examples/s]19826 examples [00:23, 889.39 examples/s]19917 examples [00:23, 894.56 examples/s]20007 examples [00:23, 841.81 examples/s]20099 examples [00:23, 863.00 examples/s]20187 examples [00:23, 866.07 examples/s]20275 examples [00:24, 865.56 examples/s]20362 examples [00:24, 852.29 examples/s]20448 examples [00:24, 852.32 examples/s]20534 examples [00:24, 834.77 examples/s]20618 examples [00:24, 817.08 examples/s]20705 examples [00:24, 828.67 examples/s]20789 examples [00:24, 827.40 examples/s]20872 examples [00:24, 802.16 examples/s]20954 examples [00:24, 806.37 examples/s]21041 examples [00:24, 822.99 examples/s]21124 examples [00:25, 810.99 examples/s]21210 examples [00:25, 823.72 examples/s]21293 examples [00:25, 817.15 examples/s]21375 examples [00:25, 798.89 examples/s]21461 examples [00:25, 813.87 examples/s]21544 examples [00:25, 816.98 examples/s]21626 examples [00:25, 814.64 examples/s]21708 examples [00:25, 814.31 examples/s]21790 examples [00:25, 805.07 examples/s]21872 examples [00:25, 807.93 examples/s]21958 examples [00:26, 821.48 examples/s]22041 examples [00:26, 822.60 examples/s]22124 examples [00:26, 808.76 examples/s]22205 examples [00:26, 797.98 examples/s]22285 examples [00:26, 777.58 examples/s]22367 examples [00:26, 787.52 examples/s]22453 examples [00:26, 805.52 examples/s]22534 examples [00:26, 803.32 examples/s]22615 examples [00:26, 764.88 examples/s]22692 examples [00:27, 752.27 examples/s]22774 examples [00:27, 769.56 examples/s]22856 examples [00:27, 781.66 examples/s]22935 examples [00:27, 775.92 examples/s]23013 examples [00:27, 760.03 examples/s]23095 examples [00:27, 775.54 examples/s]23173 examples [00:27, 755.97 examples/s]23260 examples [00:27, 785.39 examples/s]23348 examples [00:27, 810.99 examples/s]23434 examples [00:27, 824.75 examples/s]23517 examples [00:28, 810.29 examples/s]23603 examples [00:28, 823.95 examples/s]23687 examples [00:28, 828.37 examples/s]23772 examples [00:28, 833.86 examples/s]23856 examples [00:28, 827.86 examples/s]23945 examples [00:28, 844.50 examples/s]24030 examples [00:28, 825.25 examples/s]24117 examples [00:28, 837.45 examples/s]24201 examples [00:28, 802.61 examples/s]24282 examples [00:28, 798.54 examples/s]24363 examples [00:29, 800.87 examples/s]24454 examples [00:29, 828.64 examples/s]24538 examples [00:29, 831.48 examples/s]24625 examples [00:29, 841.54 examples/s]24711 examples [00:29, 844.26 examples/s]24797 examples [00:29, 847.86 examples/s]24884 examples [00:29, 852.93 examples/s]24970 examples [00:29, 841.06 examples/s]25055 examples [00:29, 801.38 examples/s]25142 examples [00:30, 819.72 examples/s]25234 examples [00:30, 845.29 examples/s]25320 examples [00:30, 829.94 examples/s]25404 examples [00:30, 831.07 examples/s]25493 examples [00:30, 845.38 examples/s]25580 examples [00:30, 851.64 examples/s]25668 examples [00:30, 859.58 examples/s]25758 examples [00:30, 871.30 examples/s]25846 examples [00:30, 869.72 examples/s]25935 examples [00:30, 874.74 examples/s]26025 examples [00:31, 880.07 examples/s]26114 examples [00:31, 878.14 examples/s]26202 examples [00:31, 876.83 examples/s]26290 examples [00:31, 877.15 examples/s]26379 examples [00:31, 880.37 examples/s]26468 examples [00:31, 882.61 examples/s]26557 examples [00:31, 870.30 examples/s]26645 examples [00:31, 868.25 examples/s]26734 examples [00:31, 873.97 examples/s]26826 examples [00:31, 885.67 examples/s]26915 examples [00:32, 883.69 examples/s]27004 examples [00:32, 874.16 examples/s]27092 examples [00:32, 866.07 examples/s]27183 examples [00:32, 862.93 examples/s]27275 examples [00:32, 877.00 examples/s]27363 examples [00:32, 870.34 examples/s]27451 examples [00:32, 862.98 examples/s]27539 examples [00:32, 865.55 examples/s]27629 examples [00:32, 874.02 examples/s]27717 examples [00:32, 874.42 examples/s]27805 examples [00:33, 872.43 examples/s]27897 examples [00:33, 885.61 examples/s]27986 examples [00:33, 878.22 examples/s]28077 examples [00:33, 885.49 examples/s]28170 examples [00:33, 897.87 examples/s]28260 examples [00:33, 891.23 examples/s]28350 examples [00:33, 877.59 examples/s]28442 examples [00:33, 889.67 examples/s]28532 examples [00:33, 844.03 examples/s]28617 examples [00:34, 823.98 examples/s]28700 examples [00:34, 759.78 examples/s]28778 examples [00:34, 742.67 examples/s]28854 examples [00:34, 747.74 examples/s]28930 examples [00:34, 736.83 examples/s]29009 examples [00:34, 750.62 examples/s]29091 examples [00:34, 767.94 examples/s]29170 examples [00:34, 771.89 examples/s]29249 examples [00:34, 776.08 examples/s]29327 examples [00:34, 769.75 examples/s]29405 examples [00:35, 767.53 examples/s]29494 examples [00:35, 800.32 examples/s]29580 examples [00:35, 816.31 examples/s]29667 examples [00:35, 830.55 examples/s]29751 examples [00:35, 831.87 examples/s]29835 examples [00:35, 818.35 examples/s]29923 examples [00:35, 835.91 examples/s]30007 examples [00:35, 775.58 examples/s]30087 examples [00:35, 780.13 examples/s]30166 examples [00:36, 781.91 examples/s]30254 examples [00:36, 807.02 examples/s]30342 examples [00:36, 826.92 examples/s]30431 examples [00:36, 843.24 examples/s]30516 examples [00:36, 809.37 examples/s]30604 examples [00:36, 827.92 examples/s]30688 examples [00:36, 826.70 examples/s]30778 examples [00:36, 845.20 examples/s]30866 examples [00:36, 852.53 examples/s]30952 examples [00:36, 846.07 examples/s]31039 examples [00:37, 850.94 examples/s]31132 examples [00:37, 873.20 examples/s]31220 examples [00:37, 872.90 examples/s]31311 examples [00:37, 880.72 examples/s]31400 examples [00:37, 835.37 examples/s]31485 examples [00:37, 830.96 examples/s]31571 examples [00:37, 838.35 examples/s]31662 examples [00:37, 856.74 examples/s]31748 examples [00:37, 852.48 examples/s]31834 examples [00:37, 840.22 examples/s]31919 examples [00:38, 825.91 examples/s]32007 examples [00:38, 840.35 examples/s]32092 examples [00:38, 780.16 examples/s]32178 examples [00:38, 799.93 examples/s]32265 examples [00:38, 817.99 examples/s]32356 examples [00:38, 843.51 examples/s]32448 examples [00:38, 863.40 examples/s]32535 examples [00:38, 850.60 examples/s]32623 examples [00:38, 856.57 examples/s]32709 examples [00:39, 818.21 examples/s]32792 examples [00:39, 811.62 examples/s]32886 examples [00:39, 845.77 examples/s]32973 examples [00:39, 851.62 examples/s]33065 examples [00:39, 869.86 examples/s]33155 examples [00:39, 875.97 examples/s]33243 examples [00:39, 847.84 examples/s]33333 examples [00:39, 861.73 examples/s]33422 examples [00:39, 868.74 examples/s]33513 examples [00:39, 878.79 examples/s]33602 examples [00:40, 876.94 examples/s]33692 examples [00:40, 880.55 examples/s]33781 examples [00:40, 855.53 examples/s]33871 examples [00:40, 866.95 examples/s]33962 examples [00:40, 877.24 examples/s]34050 examples [00:40, 865.13 examples/s]34137 examples [00:40, 865.64 examples/s]34224 examples [00:40, 863.42 examples/s]34317 examples [00:40, 880.19 examples/s]34406 examples [00:40, 868.89 examples/s]34494 examples [00:41, 840.35 examples/s]34584 examples [00:41, 855.80 examples/s]34670 examples [00:41, 832.51 examples/s]34754 examples [00:41, 829.47 examples/s]34839 examples [00:41, 833.48 examples/s]34925 examples [00:41, 840.08 examples/s]35017 examples [00:41, 860.64 examples/s]35104 examples [00:41, 837.83 examples/s]35189 examples [00:41, 822.39 examples/s]35276 examples [00:42, 835.78 examples/s]35365 examples [00:42, 849.32 examples/s]35454 examples [00:42, 859.40 examples/s]35541 examples [00:42, 857.19 examples/s]35627 examples [00:42, 825.63 examples/s]35715 examples [00:42, 838.51 examples/s]35800 examples [00:42, 839.65 examples/s]35885 examples [00:42, 827.93 examples/s]35968 examples [00:42, 805.46 examples/s]36049 examples [00:42, 805.21 examples/s]36138 examples [00:43, 827.04 examples/s]36228 examples [00:43, 846.85 examples/s]36313 examples [00:43, 836.28 examples/s]36400 examples [00:43, 844.26 examples/s]36485 examples [00:43, 839.91 examples/s]36571 examples [00:43, 845.31 examples/s]36659 examples [00:43, 855.05 examples/s]36749 examples [00:43, 866.30 examples/s]36838 examples [00:43, 873.13 examples/s]36926 examples [00:43, 851.45 examples/s]37012 examples [00:44, 843.71 examples/s]37098 examples [00:44, 848.00 examples/s]37188 examples [00:44, 860.56 examples/s]37277 examples [00:44, 866.68 examples/s]37364 examples [00:44, 847.21 examples/s]37452 examples [00:44, 855.64 examples/s]37542 examples [00:44, 865.87 examples/s]37629 examples [00:44, 861.55 examples/s]37723 examples [00:44, 881.37 examples/s]37813 examples [00:45, 886.53 examples/s]37904 examples [00:45, 892.44 examples/s]37995 examples [00:45, 894.94 examples/s]38085 examples [00:45, 867.50 examples/s]38176 examples [00:45, 878.31 examples/s]38265 examples [00:45, 873.69 examples/s]38353 examples [00:45, 860.82 examples/s]38440 examples [00:45, 860.44 examples/s]38527 examples [00:45, 842.03 examples/s]38619 examples [00:45, 861.14 examples/s]38713 examples [00:46, 881.87 examples/s]38802 examples [00:46, 863.88 examples/s]38890 examples [00:46, 866.85 examples/s]38977 examples [00:46, 864.31 examples/s]39064 examples [00:46, 864.45 examples/s]39151 examples [00:46, 862.00 examples/s]39238 examples [00:46, 856.93 examples/s]39324 examples [00:46, 856.42 examples/s]39410 examples [00:46, 838.45 examples/s]39496 examples [00:46, 842.53 examples/s]39584 examples [00:47, 852.44 examples/s]39672 examples [00:47, 859.62 examples/s]39763 examples [00:47, 873.82 examples/s]39851 examples [00:47, 868.93 examples/s]39939 examples [00:47, 870.00 examples/s]40027 examples [00:47, 796.31 examples/s]40108 examples [00:47, 799.16 examples/s]40193 examples [00:47, 810.51 examples/s]40279 examples [00:47, 824.46 examples/s]40369 examples [00:47, 844.96 examples/s]40454 examples [00:48, 818.39 examples/s]40537 examples [00:48, 821.24 examples/s]40621 examples [00:48, 824.26 examples/s]40710 examples [00:48, 841.40 examples/s]40795 examples [00:48, 821.40 examples/s]40879 examples [00:48, 826.62 examples/s]40964 examples [00:48, 832.11 examples/s]41051 examples [00:48, 841.39 examples/s]41141 examples [00:48, 855.69 examples/s]41227 examples [00:49, 855.25 examples/s]41313 examples [00:49, 832.79 examples/s]41399 examples [00:49, 840.16 examples/s]41484 examples [00:49, 833.15 examples/s]41571 examples [00:49, 843.42 examples/s]41659 examples [00:49, 852.84 examples/s]41745 examples [00:49, 847.86 examples/s]41830 examples [00:49, 805.89 examples/s]41916 examples [00:49, 820.21 examples/s]42006 examples [00:49, 841.14 examples/s]42095 examples [00:50, 852.99 examples/s]42181 examples [00:50, 852.77 examples/s]42267 examples [00:50, 836.09 examples/s]42352 examples [00:50, 839.31 examples/s]42437 examples [00:50, 839.77 examples/s]42523 examples [00:50, 843.41 examples/s]42611 examples [00:50, 848.59 examples/s]42696 examples [00:50, 837.86 examples/s]42784 examples [00:50, 849.35 examples/s]42870 examples [00:50, 833.44 examples/s]42956 examples [00:51, 840.80 examples/s]43044 examples [00:51, 849.94 examples/s]43130 examples [00:51, 777.70 examples/s]43210 examples [00:51, 746.06 examples/s]43294 examples [00:51, 770.44 examples/s]43378 examples [00:51, 789.27 examples/s]43462 examples [00:51, 801.77 examples/s]43547 examples [00:51, 813.68 examples/s]43632 examples [00:51, 823.89 examples/s]43722 examples [00:52, 844.41 examples/s]43808 examples [00:52, 846.57 examples/s]43893 examples [00:52, 841.36 examples/s]43978 examples [00:52, 793.72 examples/s]44059 examples [00:52, 781.49 examples/s]44147 examples [00:52, 807.80 examples/s]44234 examples [00:52, 823.78 examples/s]44317 examples [00:52, 802.47 examples/s]44398 examples [00:52, 802.81 examples/s]44484 examples [00:52, 817.08 examples/s]44571 examples [00:53, 831.68 examples/s]44657 examples [00:53, 839.73 examples/s]44744 examples [00:53, 846.28 examples/s]44829 examples [00:53, 835.25 examples/s]44913 examples [00:53, 818.95 examples/s]44996 examples [00:53, 789.78 examples/s]45082 examples [00:53, 808.04 examples/s]45166 examples [00:53, 815.22 examples/s]45250 examples [00:53, 821.54 examples/s]45333 examples [00:54, 816.29 examples/s]45415 examples [00:54, 806.79 examples/s]45498 examples [00:54, 812.61 examples/s]45582 examples [00:54, 820.23 examples/s]45665 examples [00:54, 780.48 examples/s]45744 examples [00:54, 762.07 examples/s]45827 examples [00:54, 780.12 examples/s]45910 examples [00:54, 794.02 examples/s]45990 examples [00:54, 776.63 examples/s]46068 examples [00:54, 776.92 examples/s]46146 examples [00:55, 774.55 examples/s]46228 examples [00:55, 785.59 examples/s]46312 examples [00:55, 799.83 examples/s]46395 examples [00:55, 807.30 examples/s]46482 examples [00:55, 822.25 examples/s]46568 examples [00:55, 830.59 examples/s]46653 examples [00:55, 836.05 examples/s]46744 examples [00:55, 854.42 examples/s]46830 examples [00:55, 842.76 examples/s]46915 examples [00:55, 830.81 examples/s]46999 examples [00:56, 819.42 examples/s]47083 examples [00:56, 823.58 examples/s]47166 examples [00:56, 814.37 examples/s]47248 examples [00:56, 799.99 examples/s]47331 examples [00:56, 807.37 examples/s]47412 examples [00:56, 771.00 examples/s]47494 examples [00:56, 783.27 examples/s]47577 examples [00:56, 793.92 examples/s]47657 examples [00:56, 788.11 examples/s]47740 examples [00:57, 797.89 examples/s]47820 examples [00:57, 794.65 examples/s]47903 examples [00:57, 803.45 examples/s]47988 examples [00:57, 814.27 examples/s]48070 examples [00:57, 803.94 examples/s]48155 examples [00:57, 816.40 examples/s]48237 examples [00:57, 815.84 examples/s]48319 examples [00:57, 788.85 examples/s]48399 examples [00:57, 765.14 examples/s]48476 examples [00:57, 759.79 examples/s]48556 examples [00:58, 771.06 examples/s]48636 examples [00:58, 777.06 examples/s]48714 examples [00:58, 759.00 examples/s]48793 examples [00:58, 767.40 examples/s]48871 examples [00:58, 769.63 examples/s]48953 examples [00:58, 781.84 examples/s]49032 examples [00:58, 783.08 examples/s]49111 examples [00:58, 776.76 examples/s]49199 examples [00:58, 802.83 examples/s]49280 examples [00:58, 800.17 examples/s]49364 examples [00:59, 811.07 examples/s]49447 examples [00:59, 815.08 examples/s]49531 examples [00:59, 818.28 examples/s]49615 examples [00:59, 823.10 examples/s]49699 examples [00:59, 826.06 examples/s]49782 examples [00:59, 809.14 examples/s]49864 examples [00:59, 812.20 examples/s]49946 examples [00:59, 801.66 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 10%|█         | 5077/50000 [00:00<00:00, 50765.93 examples/s] 31%|███       | 15369/50000 [00:00<00:00, 59867.03 examples/s] 52%|█████▏    | 25986/50000 [00:00<00:00, 68877.88 examples/s] 74%|███████▍  | 36990/50000 [00:00<00:00, 77582.97 examples/s] 94%|█████████▍| 46967/50000 [00:00<00:00, 83127.26 examples/s]                                                               0 examples [00:00, ? examples/s]64 examples [00:00, 633.33 examples/s]147 examples [00:00, 681.06 examples/s]234 examples [00:00, 727.54 examples/s]320 examples [00:00, 762.71 examples/s]408 examples [00:00, 793.72 examples/s]492 examples [00:00, 804.61 examples/s]580 examples [00:00, 823.36 examples/s]671 examples [00:00, 845.48 examples/s]759 examples [00:00, 855.43 examples/s]843 examples [00:01, 848.93 examples/s]927 examples [00:01, 834.48 examples/s]1015 examples [00:01, 846.43 examples/s]1103 examples [00:01, 855.78 examples/s]1189 examples [00:01, 853.95 examples/s]1278 examples [00:01, 862.33 examples/s]1368 examples [00:01, 872.36 examples/s]1456 examples [00:01, 871.02 examples/s]1544 examples [00:01, 871.21 examples/s]1632 examples [00:01, 860.79 examples/s]1721 examples [00:02, 866.37 examples/s]1808 examples [00:02, 843.88 examples/s]1894 examples [00:02, 847.89 examples/s]1982 examples [00:02, 855.76 examples/s]2068 examples [00:02, 852.68 examples/s]2160 examples [00:02, 870.42 examples/s]2248 examples [00:02, 871.11 examples/s]2339 examples [00:02, 882.05 examples/s]2428 examples [00:02, 881.06 examples/s]2517 examples [00:02, 850.97 examples/s]2606 examples [00:03, 860.18 examples/s]2696 examples [00:03, 871.08 examples/s]2784 examples [00:03, 870.85 examples/s]2872 examples [00:03, 843.77 examples/s]2961 examples [00:03, 855.82 examples/s]3047 examples [00:03, 850.31 examples/s]3133 examples [00:03, 839.79 examples/s]3218 examples [00:03, 826.91 examples/s]3301 examples [00:03, 801.37 examples/s]3384 examples [00:03, 807.48 examples/s]3465 examples [00:04, 806.12 examples/s]3554 examples [00:04, 828.76 examples/s]3638 examples [00:04, 832.09 examples/s]3722 examples [00:04, 820.83 examples/s]3805 examples [00:04, 797.34 examples/s]3890 examples [00:04, 811.88 examples/s]3972 examples [00:04, 784.23 examples/s]4052 examples [00:04, 788.03 examples/s]4139 examples [00:04, 809.48 examples/s]4225 examples [00:05, 821.81 examples/s]4314 examples [00:05, 838.08 examples/s]4401 examples [00:05, 846.83 examples/s]4486 examples [00:05, 845.21 examples/s]4573 examples [00:05, 852.18 examples/s]4659 examples [00:05, 823.88 examples/s]4742 examples [00:05, 804.97 examples/s]4826 examples [00:05, 813.53 examples/s]4908 examples [00:05, 815.26 examples/s]4990 examples [00:05, 804.86 examples/s]5071 examples [00:06, 750.74 examples/s]5156 examples [00:06, 777.93 examples/s]5243 examples [00:06, 801.05 examples/s]5331 examples [00:06, 821.98 examples/s]5418 examples [00:06, 833.52 examples/s]5502 examples [00:06, 597.45 examples/s]5585 examples [00:06, 650.91 examples/s]5662 examples [00:06, 680.96 examples/s]5737 examples [00:07, 692.65 examples/s]5819 examples [00:07, 725.23 examples/s]5903 examples [00:07, 755.88 examples/s]5982 examples [00:07, 722.07 examples/s]6062 examples [00:07, 739.41 examples/s]6140 examples [00:07, 749.96 examples/s]6225 examples [00:07, 775.64 examples/s]6304 examples [00:07, 777.99 examples/s]6392 examples [00:07, 804.04 examples/s]6474 examples [00:07, 806.15 examples/s]6556 examples [00:08, 804.49 examples/s]6643 examples [00:08, 819.60 examples/s]6733 examples [00:08, 841.10 examples/s]6822 examples [00:08, 853.53 examples/s]6909 examples [00:08, 858.39 examples/s]6996 examples [00:08, 855.84 examples/s]7082 examples [00:08, 798.26 examples/s]7165 examples [00:08, 805.40 examples/s]7248 examples [00:08, 812.24 examples/s]7336 examples [00:08, 830.73 examples/s]7420 examples [00:09, 829.43 examples/s]7504 examples [00:09, 831.84 examples/s]7588 examples [00:09, 827.49 examples/s]7671 examples [00:09, 807.01 examples/s]7754 examples [00:09, 812.62 examples/s]7840 examples [00:09, 824.57 examples/s]7926 examples [00:09, 834.70 examples/s]8010 examples [00:09, 825.41 examples/s]8093 examples [00:09, 819.87 examples/s]8181 examples [00:10, 835.33 examples/s]8268 examples [00:10, 843.58 examples/s]8356 examples [00:10, 853.43 examples/s]8442 examples [00:10, 842.42 examples/s]8528 examples [00:10, 846.14 examples/s]8614 examples [00:10, 849.50 examples/s]8704 examples [00:10, 862.24 examples/s]8792 examples [00:10, 865.88 examples/s]8879 examples [00:10, 834.89 examples/s]8963 examples [00:10, 815.26 examples/s]9049 examples [00:11, 827.81 examples/s]9133 examples [00:11, 803.53 examples/s]9223 examples [00:11, 828.32 examples/s]9310 examples [00:11, 838.93 examples/s]9395 examples [00:11, 842.14 examples/s]9480 examples [00:11, 841.03 examples/s]9565 examples [00:11, 832.05 examples/s]9649 examples [00:11, 831.62 examples/s]9733 examples [00:11, 825.74 examples/s]9819 examples [00:11, 835.05 examples/s]9905 examples [00:12, 841.32 examples/s]9992 examples [00:12, 848.40 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 96%|█████████▌| 9565/10000 [00:00<00:00, 95642.57 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0LSZ75/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incomplete0LSZ75/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f21af468488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f21af468488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f21af468488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f21ad4c8208>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f21357412b0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f21af468488> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f21af468488> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f21af468488> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f2135741160>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f21966ae160>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.71 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  7.71 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  7.71 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.47 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  6.47 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.28 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.28 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.10 file/s]2020-09-03 00:09:52.013598: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-09-03 00:09:52.017999: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-09-03 00:09:52.018175: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5611b13ccb80 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-09-03 00:09:52.018193: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
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

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:06, 148501.74it/s] 82%|████████▏ | 8167424/9912422 [00:00<00:08, 211979.83it/s]9920512it [00:00, 44925941.38it/s]                           
0it [00:00, ?it/s]32768it [00:00, 566046.64it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 149792.98it/s]1654784it [00:00, 10836901.80it/s]                         
0it [00:00, ?it/s]8192it [00:00, 123275.58it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '319a07408be40468a745dec1f1bfb50c10c1e19b', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/319a07408be40468a745dec1f1bfb50c10c1e19b

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/319a07408be40468a745dec1f1bfb50c10c1e19b

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f32c0c8d598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f32c0c8d598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f332c2540d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f332c2540d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f3345fa6ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f3345fa6ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f32d9ace8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f32d9ace8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f32d9ace8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {'fixed_size': 256, 'path': 'dataset/vision/MNIST/'}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 257, in compute
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 257, in compute
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
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 257, in compute
    obj_preprocessor = preprocessor_func(**args, data_info=self.data_info)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/preprocess/generic.py", line 607, in __init__
    data            = np.load( file_path,**args.get("numpy_loader_args", {}))
  File "/opt/hostedtoolcache/Python/3.6.10/x64/lib/python3.6/site-packages/numpy/lib/npyio.py", line 428, in load
    fid = open(os_fspath(file), "rb")
FileNotFoundError: [Errno 2] No such file or directory: 'train/mlmodels/dataset/text/imdb.npz'
Using TensorFlow backend.
Traceback (most recent call last):
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 445, in test_json_list
    loader.compute()
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 297, in compute
    out_tmp = preprocessor_func(input_tmp, **args)
  File "/home/runner/work/mlmodels/mlmodels/mlmodels/dataloader.py", line 92, in pickle_dump
    with open(kwargs["path"], "wb") as fi:
FileNotFoundError: [Errno 2] No such file or directory: 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:20, 122707.65it/s] 62%|██████▏   | 6193152/9912422 [00:00<00:21, 175147.52it/s]9920512it [00:00, 36496581.57it/s]                           
0it [00:00, ?it/s]32768it [00:00, 339050.71it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:11, 143718.29it/s]1654784it [00:00, 10408403.83it/s]                         
0it [00:00, ?it/s]8192it [00:00, 130416.30it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f32c0b66cc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f32c0b5ef60>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f32d9ace510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f32d9ace510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f32d9ace510> , (data_info, **args) 

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

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   9%|▊         | 14/162 [00:00<00:36,  4.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:36,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:36,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:36,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.01 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:25,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:25,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:24,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:23,  5.59 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.73 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:17,  7.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:17,  7.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:16,  7.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:16,  7.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:16,  7.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:16,  7.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:16,  7.73 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:16,  7.73 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 10.59 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:11, 10.59 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 14.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:08, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:07, 14.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:05, 18.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.22 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 24.22 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.35 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:03, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:03, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 30.35 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 36.85 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 43.42 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 49.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.18 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:01, 53.18 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 57.11 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 57.11 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.45 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 60.45 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.54 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 63.54 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 66.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.38 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 69.38 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 71.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 23.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.51s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 23.42 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.23s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:06<00:00,  3.51s/ url]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 23.42 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.23s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:06<00:00,  6.23s/ file]
Dl Size...: 100%|██████████| 162/162 [00:06<00:00, 26.01 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:06<00:00,  6.23s/ url]
0 examples [00:00, ? examples/s]2020-05-30 16:27:40.582321: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-30 16:27:40.588069: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-30 16:27:40.588239: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x561b73108f60 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-30 16:27:40.588260: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
58 examples [00:00, 574.69 examples/s]127 examples [00:00, 603.84 examples/s]209 examples [00:00, 654.69 examples/s]295 examples [00:00, 704.52 examples/s]384 examples [00:00, 750.57 examples/s]475 examples [00:00, 790.95 examples/s]570 examples [00:00, 831.39 examples/s]664 examples [00:00, 860.24 examples/s]756 examples [00:00, 876.04 examples/s]844 examples [00:01, 876.81 examples/s]932 examples [00:01, 875.94 examples/s]1027 examples [00:01, 894.97 examples/s]1121 examples [00:01, 907.71 examples/s]1215 examples [00:01, 915.51 examples/s]1307 examples [00:01, 897.70 examples/s]1397 examples [00:01, 859.26 examples/s]1484 examples [00:01, 848.48 examples/s]1570 examples [00:01, 836.28 examples/s]1665 examples [00:01, 865.01 examples/s]1758 examples [00:02, 881.18 examples/s]1850 examples [00:02, 891.09 examples/s]1944 examples [00:02, 903.77 examples/s]2038 examples [00:02, 911.80 examples/s]2130 examples [00:02, 903.88 examples/s]2221 examples [00:02, 874.94 examples/s]2311 examples [00:02, 880.27 examples/s]2405 examples [00:02, 895.42 examples/s]2495 examples [00:02, 886.39 examples/s]2589 examples [00:02, 899.90 examples/s]2680 examples [00:03, 884.10 examples/s]2769 examples [00:03, 847.26 examples/s]2855 examples [00:03, 841.68 examples/s]2940 examples [00:03, 826.56 examples/s]3023 examples [00:03, 822.35 examples/s]3106 examples [00:03, 767.49 examples/s]3184 examples [00:03, 753.94 examples/s]3270 examples [00:03, 781.22 examples/s]3349 examples [00:03, 756.47 examples/s]3432 examples [00:04, 775.18 examples/s]3518 examples [00:04, 797.83 examples/s]3606 examples [00:04, 819.44 examples/s]3699 examples [00:04, 846.93 examples/s]3786 examples [00:04, 852.18 examples/s]3872 examples [00:04, 846.83 examples/s]3957 examples [00:04, 838.41 examples/s]4049 examples [00:04, 860.33 examples/s]4136 examples [00:04, 851.60 examples/s]4225 examples [00:04, 861.56 examples/s]4313 examples [00:05, 864.60 examples/s]4400 examples [00:05, 845.13 examples/s]4493 examples [00:05, 868.34 examples/s]4585 examples [00:05, 881.34 examples/s]4674 examples [00:05, 882.56 examples/s]4767 examples [00:05, 895.79 examples/s]4860 examples [00:05, 904.24 examples/s]4951 examples [00:05, 903.69 examples/s]5042 examples [00:05, 898.69 examples/s]5132 examples [00:05, 896.94 examples/s]5222 examples [00:06, 890.11 examples/s]5312 examples [00:06, 879.59 examples/s]5401 examples [00:06, 879.56 examples/s]5490 examples [00:06, 866.13 examples/s]5577 examples [00:06, 831.17 examples/s]5661 examples [00:06, 824.32 examples/s]5753 examples [00:06, 848.45 examples/s]5840 examples [00:06, 853.97 examples/s]5932 examples [00:06, 871.70 examples/s]6020 examples [00:07, 856.68 examples/s]6106 examples [00:07, 847.29 examples/s]6194 examples [00:07, 855.73 examples/s]6287 examples [00:07, 874.31 examples/s]6379 examples [00:07, 885.93 examples/s]6468 examples [00:07, 884.23 examples/s]6557 examples [00:07, 880.38 examples/s]6648 examples [00:07, 888.52 examples/s]6737 examples [00:07, 881.97 examples/s]6826 examples [00:07, 810.24 examples/s]6911 examples [00:08, 819.71 examples/s]6994 examples [00:08, 787.59 examples/s]7080 examples [00:08, 805.80 examples/s]7169 examples [00:08, 827.37 examples/s]7256 examples [00:08, 839.39 examples/s]7344 examples [00:08, 850.75 examples/s]7430 examples [00:08, 849.82 examples/s]7520 examples [00:08, 863.23 examples/s]7611 examples [00:08, 876.31 examples/s]7699 examples [00:09, 835.19 examples/s]7786 examples [00:09, 842.25 examples/s]7871 examples [00:09, 833.38 examples/s]7959 examples [00:09, 845.50 examples/s]8044 examples [00:09, 845.55 examples/s]8129 examples [00:09, 844.84 examples/s]8219 examples [00:09, 858.04 examples/s]8306 examples [00:09, 859.80 examples/s]8393 examples [00:09, 850.48 examples/s]8483 examples [00:09, 864.20 examples/s]8577 examples [00:10, 884.83 examples/s]8671 examples [00:10, 898.35 examples/s]8762 examples [00:10, 900.41 examples/s]8853 examples [00:10, 892.22 examples/s]8945 examples [00:10, 899.24 examples/s]9036 examples [00:10, 861.92 examples/s]9123 examples [00:10, 862.23 examples/s]9213 examples [00:10, 871.63 examples/s]9302 examples [00:10, 875.98 examples/s]9390 examples [00:10, 824.66 examples/s]9480 examples [00:11, 845.85 examples/s]9566 examples [00:11, 835.67 examples/s]9651 examples [00:11, 831.85 examples/s]9735 examples [00:11, 792.52 examples/s]9819 examples [00:11, 803.12 examples/s]9910 examples [00:11, 830.20 examples/s]9994 examples [00:11, 831.51 examples/s]10078 examples [00:11, 806.35 examples/s]10162 examples [00:11, 814.53 examples/s]10245 examples [00:12, 815.77 examples/s]10327 examples [00:12, 808.84 examples/s]10409 examples [00:12, 811.58 examples/s]10492 examples [00:12, 814.45 examples/s]10576 examples [00:12, 821.78 examples/s]10661 examples [00:12, 828.75 examples/s]10744 examples [00:12, 813.02 examples/s]10826 examples [00:12, 793.92 examples/s]10906 examples [00:12, 792.06 examples/s]10986 examples [00:12, 792.64 examples/s]11066 examples [00:13, 776.27 examples/s]11148 examples [00:13, 788.16 examples/s]11227 examples [00:13, 787.62 examples/s]11314 examples [00:13, 809.65 examples/s]11407 examples [00:13, 842.14 examples/s]11492 examples [00:13, 830.64 examples/s]11579 examples [00:13, 840.17 examples/s]11668 examples [00:13, 851.26 examples/s]11755 examples [00:13, 855.86 examples/s]11844 examples [00:13, 863.71 examples/s]11931 examples [00:14, 820.23 examples/s]12014 examples [00:14, 789.75 examples/s]12094 examples [00:14, 782.97 examples/s]12180 examples [00:14, 802.07 examples/s]12264 examples [00:14, 810.72 examples/s]12346 examples [00:14, 800.90 examples/s]12427 examples [00:14, 789.96 examples/s]12518 examples [00:14, 821.62 examples/s]12601 examples [00:14, 802.98 examples/s]12684 examples [00:15, 808.95 examples/s]12777 examples [00:15, 841.05 examples/s]12862 examples [00:15, 837.31 examples/s]12950 examples [00:15, 849.03 examples/s]13037 examples [00:15, 853.08 examples/s]13132 examples [00:15, 879.41 examples/s]13222 examples [00:15, 884.74 examples/s]13311 examples [00:15, 860.85 examples/s]13404 examples [00:15, 878.62 examples/s]13493 examples [00:15, 873.00 examples/s]13581 examples [00:16, 819.51 examples/s]13666 examples [00:16, 827.64 examples/s]13750 examples [00:16, 809.92 examples/s]13838 examples [00:16, 826.99 examples/s]13922 examples [00:16, 820.70 examples/s]14005 examples [00:16, 811.18 examples/s]14090 examples [00:16, 821.74 examples/s]14173 examples [00:16, 823.86 examples/s]14259 examples [00:16, 831.92 examples/s]14346 examples [00:16, 840.49 examples/s]14431 examples [00:17, 828.21 examples/s]14517 examples [00:17, 837.23 examples/s]14601 examples [00:17, 822.30 examples/s]14687 examples [00:17, 832.86 examples/s]14771 examples [00:17, 815.74 examples/s]14863 examples [00:17, 842.90 examples/s]14949 examples [00:17, 847.18 examples/s]15034 examples [00:17, 802.80 examples/s]15115 examples [00:17, 791.88 examples/s]15209 examples [00:18, 830.74 examples/s]15304 examples [00:18, 863.10 examples/s]15395 examples [00:18, 876.44 examples/s]15488 examples [00:18, 889.26 examples/s]15578 examples [00:18, 884.18 examples/s]15667 examples [00:18, 883.17 examples/s]15756 examples [00:18, 881.58 examples/s]15845 examples [00:18, 866.87 examples/s]15939 examples [00:18, 887.11 examples/s]16031 examples [00:18, 894.97 examples/s]16121 examples [00:19, 881.19 examples/s]16210 examples [00:19, 849.56 examples/s]16297 examples [00:19, 855.58 examples/s]16383 examples [00:19, 841.23 examples/s]16468 examples [00:19, 814.58 examples/s]16553 examples [00:19, 822.24 examples/s]16638 examples [00:19, 828.29 examples/s]16726 examples [00:19, 839.41 examples/s]16821 examples [00:19, 867.43 examples/s]16909 examples [00:19, 862.87 examples/s]16999 examples [00:20, 873.35 examples/s]17087 examples [00:20, 861.37 examples/s]17181 examples [00:20, 882.38 examples/s]17270 examples [00:20, 879.29 examples/s]17359 examples [00:20, 874.46 examples/s]17452 examples [00:20, 887.76 examples/s]17541 examples [00:20, 881.98 examples/s]17636 examples [00:20, 899.34 examples/s]17731 examples [00:20, 912.97 examples/s]17823 examples [00:20, 905.91 examples/s]17914 examples [00:21, 877.25 examples/s]18003 examples [00:21, 816.30 examples/s]18086 examples [00:21, 819.26 examples/s]18169 examples [00:21, 797.23 examples/s]18250 examples [00:21, 788.46 examples/s]18338 examples [00:21, 813.68 examples/s]18432 examples [00:21, 846.09 examples/s]18522 examples [00:21, 859.70 examples/s]18609 examples [00:21, 854.27 examples/s]18695 examples [00:22, 843.86 examples/s]18782 examples [00:22, 851.11 examples/s]18868 examples [00:22, 853.13 examples/s]18954 examples [00:22, 847.51 examples/s]19040 examples [00:22, 849.06 examples/s]19125 examples [00:22, 846.75 examples/s]19210 examples [00:22, 832.92 examples/s]19304 examples [00:22, 860.15 examples/s]19401 examples [00:22, 889.77 examples/s]19494 examples [00:22, 900.13 examples/s]19586 examples [00:23, 903.90 examples/s]19677 examples [00:23, 849.82 examples/s]19773 examples [00:23, 878.92 examples/s]19869 examples [00:23, 900.76 examples/s]19963 examples [00:23, 911.17 examples/s]20055 examples [00:23, 880.07 examples/s]20150 examples [00:23, 899.77 examples/s]20242 examples [00:23, 903.92 examples/s]20334 examples [00:23, 907.51 examples/s]20426 examples [00:24, 905.04 examples/s]20517 examples [00:24, 894.08 examples/s]20607 examples [00:24, 889.62 examples/s]20697 examples [00:24, 882.36 examples/s]20786 examples [00:24, 876.90 examples/s]20874 examples [00:24, 872.30 examples/s]20966 examples [00:24, 883.51 examples/s]21055 examples [00:24, 874.26 examples/s]21143 examples [00:24, 848.34 examples/s]21229 examples [00:24, 843.92 examples/s]21314 examples [00:25, 819.22 examples/s]21397 examples [00:25, 817.65 examples/s]21479 examples [00:25, 804.39 examples/s]21568 examples [00:25, 827.27 examples/s]21654 examples [00:25, 836.72 examples/s]21739 examples [00:25, 838.10 examples/s]21830 examples [00:25, 856.05 examples/s]21924 examples [00:25, 877.14 examples/s]22012 examples [00:25, 876.53 examples/s]22106 examples [00:25, 892.24 examples/s]22196 examples [00:26, 857.41 examples/s]22283 examples [00:26, 832.23 examples/s]22367 examples [00:26, 812.79 examples/s]22450 examples [00:26, 817.42 examples/s]22540 examples [00:26, 840.24 examples/s]22627 examples [00:26, 847.36 examples/s]22713 examples [00:26, 825.21 examples/s]22796 examples [00:26, 805.74 examples/s]22877 examples [00:26, 783.82 examples/s]22957 examples [00:27, 787.16 examples/s]23036 examples [00:27, 787.70 examples/s]23119 examples [00:27, 798.99 examples/s]23210 examples [00:27, 828.43 examples/s]23303 examples [00:27, 854.15 examples/s]23389 examples [00:27, 848.66 examples/s]23475 examples [00:27, 834.60 examples/s]23561 examples [00:27, 841.85 examples/s]23652 examples [00:27, 859.17 examples/s]23740 examples [00:27, 863.86 examples/s]23827 examples [00:28, 851.96 examples/s]23915 examples [00:28, 858.70 examples/s]24001 examples [00:28, 826.77 examples/s]24085 examples [00:28, 802.21 examples/s]24168 examples [00:28, 809.37 examples/s]24250 examples [00:28, 801.85 examples/s]24338 examples [00:28, 823.69 examples/s]24421 examples [00:28, 813.62 examples/s]24504 examples [00:28, 818.06 examples/s]24586 examples [00:29, 792.43 examples/s]24666 examples [00:29, 769.26 examples/s]24744 examples [00:29, 737.86 examples/s]24819 examples [00:29, 719.76 examples/s]24905 examples [00:29, 756.52 examples/s]24984 examples [00:29, 764.92 examples/s]25065 examples [00:29, 776.46 examples/s]25145 examples [00:29, 780.89 examples/s]25229 examples [00:29, 795.64 examples/s]25313 examples [00:29, 807.99 examples/s]25398 examples [00:30, 817.84 examples/s]25480 examples [00:30, 816.13 examples/s]25564 examples [00:30, 822.25 examples/s]25649 examples [00:30, 829.79 examples/s]25733 examples [00:30, 816.07 examples/s]25822 examples [00:30, 836.25 examples/s]25906 examples [00:30, 833.26 examples/s]25994 examples [00:30, 844.20 examples/s]26082 examples [00:30, 854.34 examples/s]26171 examples [00:30, 862.52 examples/s]26258 examples [00:31, 860.92 examples/s]26349 examples [00:31, 875.05 examples/s]26437 examples [00:31, 862.07 examples/s]26524 examples [00:31, 837.44 examples/s]26612 examples [00:31, 847.39 examples/s]26699 examples [00:31, 853.30 examples/s]26786 examples [00:31, 856.26 examples/s]26878 examples [00:31, 874.40 examples/s]26968 examples [00:31, 880.27 examples/s]27061 examples [00:31, 894.56 examples/s]27151 examples [00:32, 891.69 examples/s]27245 examples [00:32, 905.63 examples/s]27336 examples [00:32, 902.91 examples/s]27427 examples [00:32, 893.72 examples/s]27517 examples [00:32, 886.58 examples/s]27607 examples [00:32, 888.31 examples/s]27696 examples [00:32, 863.24 examples/s]27787 examples [00:32, 874.52 examples/s]27875 examples [00:32, 865.86 examples/s]27964 examples [00:33, 870.57 examples/s]28056 examples [00:33, 884.06 examples/s]28145 examples [00:33, 840.13 examples/s]28230 examples [00:33, 834.05 examples/s]28316 examples [00:33, 836.00 examples/s]28400 examples [00:33, 804.17 examples/s]28481 examples [00:33, 800.78 examples/s]28562 examples [00:33, 780.10 examples/s]28644 examples [00:33, 790.10 examples/s]28724 examples [00:33, 789.29 examples/s]28812 examples [00:34, 814.09 examples/s]28894 examples [00:34, 815.46 examples/s]28976 examples [00:34, 812.10 examples/s]29064 examples [00:34, 830.92 examples/s]29148 examples [00:34, 827.03 examples/s]29231 examples [00:34, 797.07 examples/s]29312 examples [00:34, 796.35 examples/s]29405 examples [00:34, 831.64 examples/s]29500 examples [00:34, 862.46 examples/s]29591 examples [00:34, 875.57 examples/s]29681 examples [00:35, 880.36 examples/s]29772 examples [00:35, 886.49 examples/s]29862 examples [00:35, 887.90 examples/s]29951 examples [00:35, 877.93 examples/s]30039 examples [00:35, 798.95 examples/s]30129 examples [00:35, 825.09 examples/s]30214 examples [00:35, 831.62 examples/s]30299 examples [00:35, 820.60 examples/s]30382 examples [00:35, 804.19 examples/s]30465 examples [00:36, 811.39 examples/s]30547 examples [00:36, 797.41 examples/s]30640 examples [00:36, 831.54 examples/s]30729 examples [00:36, 847.59 examples/s]30820 examples [00:36, 859.93 examples/s]30907 examples [00:36, 857.90 examples/s]30994 examples [00:36, 849.70 examples/s]31080 examples [00:36, 804.56 examples/s]31168 examples [00:36, 823.67 examples/s]31257 examples [00:36, 839.64 examples/s]31342 examples [00:37, 821.18 examples/s]31425 examples [00:37, 795.51 examples/s]31506 examples [00:37, 781.13 examples/s]31587 examples [00:37, 788.45 examples/s]31673 examples [00:37, 806.54 examples/s]31759 examples [00:37, 820.13 examples/s]31843 examples [00:37, 825.80 examples/s]31936 examples [00:37, 852.70 examples/s]32026 examples [00:37, 865.15 examples/s]32113 examples [00:38, 843.40 examples/s]32206 examples [00:38, 867.54 examples/s]32299 examples [00:38, 882.36 examples/s]32389 examples [00:38, 885.58 examples/s]32480 examples [00:38, 891.26 examples/s]32570 examples [00:38, 876.02 examples/s]32666 examples [00:38, 897.28 examples/s]32759 examples [00:38, 904.57 examples/s]32850 examples [00:38, 897.17 examples/s]32942 examples [00:38, 902.44 examples/s]33033 examples [00:39, 873.93 examples/s]33126 examples [00:39, 888.00 examples/s]33216 examples [00:39, 883.17 examples/s]33306 examples [00:39, 886.85 examples/s]33399 examples [00:39, 898.08 examples/s]33489 examples [00:39, 892.26 examples/s]33579 examples [00:39, 890.85 examples/s]33669 examples [00:39, 890.66 examples/s]33763 examples [00:39, 902.53 examples/s]33854 examples [00:39, 895.60 examples/s]33946 examples [00:40, 900.93 examples/s]34039 examples [00:40, 907.48 examples/s]34130 examples [00:40, 885.78 examples/s]34219 examples [00:40, 862.42 examples/s]34306 examples [00:40, 861.36 examples/s]34393 examples [00:40, 852.66 examples/s]34479 examples [00:40, 822.82 examples/s]34562 examples [00:40, 820.01 examples/s]34650 examples [00:40, 836.15 examples/s]34735 examples [00:41, 837.89 examples/s]34823 examples [00:41, 848.09 examples/s]34910 examples [00:41, 852.16 examples/s]35002 examples [00:41, 871.02 examples/s]35099 examples [00:41, 896.08 examples/s]35191 examples [00:41, 900.00 examples/s]35282 examples [00:41, 875.97 examples/s]35370 examples [00:41, 824.43 examples/s]35461 examples [00:41, 847.01 examples/s]35554 examples [00:41, 868.43 examples/s]35649 examples [00:42, 890.11 examples/s]35739 examples [00:42, 859.26 examples/s]35826 examples [00:42, 861.19 examples/s]35913 examples [00:42, 861.07 examples/s]36000 examples [00:42, 860.10 examples/s]36094 examples [00:42, 881.36 examples/s]36188 examples [00:42, 898.01 examples/s]36279 examples [00:42, 895.20 examples/s]36369 examples [00:42, 894.99 examples/s]36459 examples [00:42, 891.57 examples/s]36549 examples [00:43, 882.28 examples/s]36638 examples [00:43, 853.32 examples/s]36724 examples [00:43, 849.59 examples/s]36811 examples [00:43, 854.75 examples/s]36907 examples [00:43, 883.74 examples/s]36997 examples [00:43, 886.60 examples/s]37086 examples [00:43, 878.01 examples/s]37178 examples [00:43, 889.45 examples/s]37270 examples [00:43, 896.80 examples/s]37362 examples [00:43, 903.47 examples/s]37453 examples [00:44, 899.92 examples/s]37544 examples [00:44, 883.88 examples/s]37633 examples [00:44, 860.29 examples/s]37720 examples [00:44, 846.63 examples/s]37805 examples [00:44, 817.11 examples/s]37888 examples [00:44, 806.55 examples/s]37969 examples [00:44, 806.40 examples/s]38051 examples [00:44, 810.34 examples/s]38139 examples [00:44, 828.90 examples/s]38227 examples [00:45, 842.54 examples/s]38314 examples [00:45, 848.88 examples/s]38401 examples [00:45, 854.44 examples/s]38487 examples [00:45, 816.14 examples/s]38572 examples [00:45, 824.02 examples/s]38655 examples [00:45, 824.95 examples/s]38747 examples [00:45, 849.67 examples/s]38841 examples [00:45, 872.49 examples/s]38929 examples [00:45, 866.21 examples/s]39016 examples [00:45, 864.10 examples/s]39103 examples [00:46, 860.53 examples/s]39190 examples [00:46, 860.10 examples/s]39279 examples [00:46, 866.39 examples/s]39372 examples [00:46, 884.27 examples/s]39468 examples [00:46, 904.31 examples/s]39562 examples [00:46, 914.44 examples/s]39654 examples [00:46, 909.77 examples/s]39750 examples [00:46, 921.68 examples/s]39843 examples [00:46, 921.40 examples/s]39936 examples [00:46, 909.86 examples/s]40028 examples [00:47, 861.99 examples/s]40124 examples [00:47, 887.76 examples/s]40217 examples [00:47, 898.76 examples/s]40308 examples [00:47, 883.54 examples/s]40397 examples [00:47, 854.99 examples/s]40483 examples [00:47, 849.95 examples/s]40569 examples [00:47, 790.95 examples/s]40650 examples [00:47, 776.12 examples/s]40731 examples [00:47, 784.04 examples/s]40814 examples [00:48, 796.49 examples/s]40901 examples [00:48, 813.93 examples/s]40983 examples [00:48, 803.91 examples/s]41068 examples [00:48, 817.10 examples/s]41150 examples [00:48, 811.63 examples/s]41232 examples [00:48, 786.89 examples/s]41312 examples [00:48, 790.46 examples/s]41392 examples [00:48, 757.67 examples/s]41481 examples [00:48, 792.57 examples/s]41576 examples [00:48, 833.43 examples/s]41671 examples [00:49, 863.52 examples/s]41759 examples [00:49, 849.48 examples/s]41851 examples [00:49, 868.14 examples/s]41939 examples [00:49, 867.39 examples/s]42027 examples [00:49, 869.15 examples/s]42115 examples [00:49, 869.81 examples/s]42203 examples [00:49, 848.45 examples/s]42289 examples [00:49, 827.66 examples/s]42381 examples [00:49, 852.16 examples/s]42469 examples [00:50, 860.26 examples/s]42556 examples [00:50, 860.71 examples/s]42643 examples [00:50, 861.14 examples/s]42730 examples [00:50, 858.61 examples/s]42816 examples [00:50, 843.95 examples/s]42903 examples [00:50, 850.73 examples/s]42989 examples [00:50, 851.80 examples/s]43075 examples [00:50, 827.97 examples/s]43158 examples [00:50, 777.27 examples/s]43239 examples [00:50, 786.53 examples/s]43321 examples [00:51, 795.01 examples/s]43403 examples [00:51, 800.40 examples/s]43489 examples [00:51, 815.53 examples/s]43577 examples [00:51, 833.41 examples/s]43662 examples [00:51, 835.67 examples/s]43749 examples [00:51, 843.90 examples/s]43834 examples [00:51, 844.76 examples/s]43919 examples [00:51, 832.48 examples/s]44003 examples [00:51, 820.83 examples/s]44092 examples [00:51, 839.07 examples/s]44183 examples [00:52, 858.67 examples/s]44270 examples [00:52, 860.73 examples/s]44360 examples [00:52, 871.95 examples/s]44448 examples [00:52, 831.49 examples/s]44532 examples [00:52, 785.05 examples/s]44612 examples [00:52, 767.59 examples/s]44694 examples [00:52, 780.92 examples/s]44784 examples [00:52, 812.82 examples/s]44875 examples [00:52, 838.47 examples/s]44964 examples [00:53, 838.29 examples/s]45057 examples [00:53, 861.78 examples/s]45149 examples [00:53, 877.16 examples/s]45238 examples [00:53, 867.00 examples/s]45333 examples [00:53, 889.99 examples/s]45426 examples [00:53, 900.48 examples/s]45517 examples [00:53, 866.17 examples/s]45605 examples [00:53, 861.65 examples/s]45692 examples [00:53, 836.06 examples/s]45777 examples [00:53, 802.37 examples/s]45860 examples [00:54, 807.99 examples/s]45944 examples [00:54, 815.36 examples/s]46033 examples [00:54, 834.97 examples/s]46121 examples [00:54, 846.65 examples/s]46211 examples [00:54, 861.12 examples/s]46298 examples [00:54, 828.65 examples/s]46382 examples [00:54, 817.42 examples/s]46465 examples [00:54, 777.06 examples/s]46544 examples [00:54, 779.46 examples/s]46627 examples [00:55, 792.65 examples/s]46715 examples [00:55, 816.20 examples/s]46800 examples [00:55, 825.13 examples/s]46883 examples [00:55, 825.13 examples/s]46967 examples [00:55, 828.33 examples/s]47050 examples [00:55, 820.51 examples/s]47142 examples [00:55, 847.79 examples/s]47235 examples [00:55, 869.20 examples/s]47327 examples [00:55, 881.76 examples/s]47416 examples [00:55, 854.92 examples/s]47502 examples [00:56, 819.08 examples/s]47593 examples [00:56, 844.23 examples/s]47681 examples [00:56, 852.42 examples/s]47774 examples [00:56, 872.98 examples/s]47866 examples [00:56, 885.91 examples/s]47958 examples [00:56, 893.85 examples/s]48050 examples [00:56, 900.90 examples/s]48143 examples [00:56, 907.13 examples/s]48237 examples [00:56, 915.59 examples/s]48329 examples [00:56, 891.54 examples/s]48420 examples [00:57, 896.85 examples/s]48510 examples [00:57, 890.86 examples/s]48601 examples [00:57, 894.78 examples/s]48693 examples [00:57, 900.41 examples/s]48787 examples [00:57, 911.66 examples/s]48879 examples [00:57, 886.33 examples/s]48968 examples [00:57, 886.40 examples/s]49057 examples [00:57, 855.72 examples/s]49145 examples [00:57, 862.10 examples/s]49232 examples [00:58, 863.73 examples/s]49320 examples [00:58, 868.25 examples/s]49407 examples [00:58, 866.19 examples/s]49494 examples [00:58, 864.14 examples/s]49588 examples [00:58, 882.80 examples/s]49677 examples [00:58, 844.11 examples/s]49763 examples [00:58, 848.06 examples/s]49849 examples [00:58, 841.92 examples/s]49934 examples [00:58, 805.16 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s]  8%|▊         | 3940/50000 [00:00<00:01, 39399.19 examples/s] 26%|██▋       | 13163/50000 [00:00<00:00, 47574.49 examples/s] 44%|████▍     | 22240/50000 [00:00<00:00, 55496.46 examples/s] 61%|██████    | 30515/50000 [00:00<00:00, 61579.86 examples/s] 82%|████████▏ | 40753/50000 [00:00<00:00, 69941.66 examples/s]                                                               0 examples [00:00, ? examples/s]68 examples [00:00, 678.09 examples/s]142 examples [00:00, 695.42 examples/s]220 examples [00:00, 718.05 examples/s]307 examples [00:00, 756.48 examples/s]379 examples [00:00, 743.27 examples/s]450 examples [00:00, 732.66 examples/s]543 examples [00:00, 781.86 examples/s]639 examples [00:00, 827.65 examples/s]733 examples [00:00, 857.16 examples/s]827 examples [00:01, 880.02 examples/s]921 examples [00:01, 896.19 examples/s]1015 examples [00:01, 907.45 examples/s]1106 examples [00:01, 899.74 examples/s]1198 examples [00:01, 903.19 examples/s]1289 examples [00:01, 892.12 examples/s]1379 examples [00:01, 711.26 examples/s]1471 examples [00:01, 761.51 examples/s]1553 examples [00:01, 777.30 examples/s]1635 examples [00:02, 782.30 examples/s]1716 examples [00:02, 785.35 examples/s]1798 examples [00:02, 791.97 examples/s]1883 examples [00:02, 807.46 examples/s]1975 examples [00:02, 837.49 examples/s]2060 examples [00:02, 840.82 examples/s]2149 examples [00:02, 853.48 examples/s]2235 examples [00:02, 841.10 examples/s]2320 examples [00:02, 826.15 examples/s]2408 examples [00:02, 840.48 examples/s]2499 examples [00:03, 859.57 examples/s]2588 examples [00:03, 867.50 examples/s]2681 examples [00:03, 884.98 examples/s]2773 examples [00:03, 894.27 examples/s]2863 examples [00:03, 893.78 examples/s]2953 examples [00:03, 834.93 examples/s]3038 examples [00:03, 825.73 examples/s]3123 examples [00:03, 828.78 examples/s]3214 examples [00:03, 850.14 examples/s]3301 examples [00:03, 852.31 examples/s]3387 examples [00:04, 837.16 examples/s]3479 examples [00:04, 858.93 examples/s]3570 examples [00:04, 872.86 examples/s]3661 examples [00:04, 883.46 examples/s]3754 examples [00:04, 895.06 examples/s]3844 examples [00:04, 883.04 examples/s]3933 examples [00:04, 884.52 examples/s]4028 examples [00:04, 903.18 examples/s]4122 examples [00:04, 911.72 examples/s]4214 examples [00:04, 867.88 examples/s]4302 examples [00:05, 843.18 examples/s]4387 examples [00:05, 816.44 examples/s]4472 examples [00:05, 823.98 examples/s]4564 examples [00:05, 849.82 examples/s]4650 examples [00:05, 844.02 examples/s]4735 examples [00:05, 823.74 examples/s]4818 examples [00:05, 812.13 examples/s]4900 examples [00:05, 806.85 examples/s]4982 examples [00:05, 809.74 examples/s]5064 examples [00:06, 812.18 examples/s]5152 examples [00:06, 829.76 examples/s]5236 examples [00:06, 804.91 examples/s]5317 examples [00:06, 801.43 examples/s]5400 examples [00:06, 806.94 examples/s]5484 examples [00:06, 815.24 examples/s]5566 examples [00:06, 809.02 examples/s]5657 examples [00:06, 834.59 examples/s]5752 examples [00:06, 864.37 examples/s]5839 examples [00:06, 865.96 examples/s]5926 examples [00:07, 866.70 examples/s]6013 examples [00:07, 846.56 examples/s]6098 examples [00:07, 829.15 examples/s]6182 examples [00:07, 823.90 examples/s]6265 examples [00:07, 808.37 examples/s]6355 examples [00:07, 832.09 examples/s]6443 examples [00:07, 845.80 examples/s]6531 examples [00:07, 850.98 examples/s]6618 examples [00:07, 855.14 examples/s]6704 examples [00:08, 808.52 examples/s]6790 examples [00:08, 822.85 examples/s]6873 examples [00:08, 816.51 examples/s]6959 examples [00:08, 826.19 examples/s]7042 examples [00:08, 815.91 examples/s]7124 examples [00:08, 814.30 examples/s]7214 examples [00:08, 837.82 examples/s]7306 examples [00:08, 859.33 examples/s]7393 examples [00:08, 860.23 examples/s]7482 examples [00:08, 868.08 examples/s]7569 examples [00:09, 858.79 examples/s]7658 examples [00:09, 867.70 examples/s]7752 examples [00:09, 887.51 examples/s]7841 examples [00:09, 878.59 examples/s]7930 examples [00:09, 878.98 examples/s]8019 examples [00:09, 860.77 examples/s]8106 examples [00:09, 822.81 examples/s]8195 examples [00:09, 841.41 examples/s]8285 examples [00:09, 856.93 examples/s]8378 examples [00:09, 876.93 examples/s]8470 examples [00:10, 889.39 examples/s]8562 examples [00:10, 898.13 examples/s]8655 examples [00:10, 900.10 examples/s]8746 examples [00:10, 884.30 examples/s]8840 examples [00:10, 899.91 examples/s]8933 examples [00:10, 906.41 examples/s]9024 examples [00:10, 894.92 examples/s]9119 examples [00:10, 908.21 examples/s]9214 examples [00:10, 917.96 examples/s]9309 examples [00:10, 924.95 examples/s]9404 examples [00:11, 930.32 examples/s]9498 examples [00:11, 923.81 examples/s]9591 examples [00:11, 884.35 examples/s]9680 examples [00:11, 872.51 examples/s]9768 examples [00:11, 852.80 examples/s]9854 examples [00:11, 849.69 examples/s]9940 examples [00:11, 832.03 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s] 95%|█████████▍| 9493/10000 [00:00<00:00, 94920.59 examples/s]                                                              [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUBJEPB/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteUBJEPB/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f32d9ace8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f32d9ace8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f32d9ace8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f325ee70ef0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f3278049ef0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f32d9ace8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f32d9ace8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f32d9ace8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f32bfe8f2b0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f32bfe8f048>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f32515c0268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f32515c0268> 

  function with postional parmater data_info <function split_train_valid at 0x7f32515c0268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f32515c0378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f32515c0378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f32515c0378> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:05<149:30:02, 1.60kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:05<104:52:45, 2.28kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:05<73:27:23, 3.26kB/s]  .vector_cache/glove.6B.zip:   0%|          | 901k/862M [00:05<51:23:35, 4.66kB/s].vector_cache/glove.6B.zip:   0%|          | 3.65M/862M [00:05<35:51:50, 6.65kB/s].vector_cache/glove.6B.zip:   1%|          | 8.92M/862M [00:05<24:57:06, 9.50kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.9M/862M [00:05<17:23:15, 13.6kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.3M/862M [00:05<12:06:31, 19.4kB/s].vector_cache/glove.6B.zip:   3%|▎         | 21.9M/862M [00:06<8:25:55, 27.7kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 27.6M/862M [00:06<5:51:52, 39.5kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.2M/862M [00:06<4:04:44, 56.5kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.2M/862M [00:06<2:50:22, 80.6kB/s].vector_cache/glove.6B.zip:   5%|▍         | 42.1M/862M [00:06<1:58:47, 115kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 47.0M/862M [00:06<1:22:44, 164kB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.4M/862M [00:06<57:42, 234kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 51.5M/862M [00:06<47:29, 285kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.6M/862M [00:08<34:59, 384kB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:08<26:19, 511kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.0M/862M [00:09<18:47, 714kB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.5M/862M [00:09<13:16, 1.01MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.7M/862M [00:10<45:36, 293kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 60.1M/862M [00:10<33:30, 399kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.4M/862M [00:11<23:49, 560kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:12<19:22, 687kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.2M/862M [00:12<15:00, 887kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.7M/862M [00:13<10:46, 1.23MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.1M/862M [00:14<10:30, 1.26MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:14<10:02, 1.32MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.0M/862M [00:15<07:41, 1.72MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:15<05:30, 2.39MB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.2M/862M [00:16<12:45:26, 17.2kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.6M/862M [00:16<8:56:54, 24.5kB/s] .vector_cache/glove.6B.zip:   9%|▊         | 74.1M/862M [00:16<6:15:22, 35.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.3M/862M [00:18<4:25:08, 49.4kB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.5M/862M [00:18<3:08:12, 69.6kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:18<2:12:11, 99.0kB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:19<1:32:28, 141kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:20<1:10:29, 185kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.8M/862M [00:20<50:38, 257kB/s]  .vector_cache/glove.6B.zip:  10%|▉         | 82.3M/862M [00:20<35:42, 364kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.5M/862M [00:22<27:58, 463kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.7M/862M [00:22<22:14, 583kB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.5M/862M [00:22<16:12, 799kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:24<13:23, 963kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.0M/862M [00:24<10:40, 1.21MB/s].vector_cache/glove.6B.zip:  11%|█         | 90.6M/862M [00:24<07:45, 1.66MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.7M/862M [00:26<08:25, 1.52MB/s].vector_cache/glove.6B.zip:  11%|█         | 92.9M/862M [00:26<08:30, 1.51MB/s].vector_cache/glove.6B.zip:  11%|█         | 93.7M/862M [00:26<06:29, 1.97MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.0M/862M [00:26<04:42, 2.71MB/s].vector_cache/glove.6B.zip:  11%|█         | 96.9M/862M [00:28<10:12, 1.25MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.2M/862M [00:28<08:14, 1.55MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 98.4M/862M [00:28<06:05, 2.09MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:28<04:24, 2.88MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<27:05, 468kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:30<21:32, 589kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:30<15:36, 811kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 104M/862M [00:30<11:05, 1.14MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:32<13:24, 941kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:32<10:40, 1.18MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:32<07:46, 1.62MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:34<08:22, 1.50MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:34<08:26, 1.49MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:34<06:33, 1.91MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:34<04:44, 2.64MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:36<1:32:19, 135kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<1:05:41, 190kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:36<46:23, 269kB/s]  .vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:36<32:30, 382kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:38<52:28, 237kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:38<36:42, 337kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<30:26, 405kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:40<22:33, 547kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<16:04, 766kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<14:04, 872kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:42<12:27, 985kB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<09:15, 1.32MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 129M/862M [00:42<06:36, 1.85MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:44<12:04, 1.01MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:44<09:43, 1.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:44<07:06, 1.71MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:46<07:46, 1.56MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:46<07:54, 1.53MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<06:08, 1.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:46<04:25, 2.73MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:48<11:38:11, 17.3kB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:48<8:09:43, 24.6kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:48<5:42:22, 35.2kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 142M/862M [00:50<4:01:45, 49.6kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 142M/862M [00:50<2:51:37, 69.9kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<2:00:37, 99.4kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 146M/862M [00:52<1:26:00, 139kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<1:01:22, 194kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:52<43:10, 276kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 150M/862M [00:53<32:55, 360kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 151M/862M [00:54<25:28, 465kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<18:21, 645kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 153M/862M [00:54<13:00, 908kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:55<13:23, 881kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<10:34, 1.12MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<07:37, 1.54MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:57<08:06, 1.45MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:58<06:51, 1.71MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 161M/862M [00:58<05:02, 2.32MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [00:59<06:18, 1.85MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<06:55, 1.68MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 163M/862M [01:00<05:28, 2.13MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:00<04:06, 2.83MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<05:34, 2.08MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 167M/862M [01:01<05:04, 2.28MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [01:02<03:50, 3.00MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<05:27, 2.11MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [01:03<06:11, 1.86MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<04:49, 2.39MB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<03:37, 3.17MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:05<05:38, 2.03MB/s].vector_cache/glove.6B.zip:  20%|██        | 175M/862M [01:05<05:08, 2.23MB/s].vector_cache/glove.6B.zip:  21%|██        | 177M/862M [01:06<03:50, 2.97MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:07<05:21, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 179M/862M [01:07<06:04, 1.88MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<04:43, 2.41MB/s].vector_cache/glove.6B.zip:  21%|██        | 182M/862M [01:08<03:28, 3.27MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 183M/862M [01:09<07:08, 1.58MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:09<06:09, 1.84MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:09<04:35, 2.46MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 187M/862M [01:11<05:50, 1.92MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<06:22, 1.76MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:11<05:02, 2.23MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:12<03:39, 3.06MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 191M/862M [01:13<1:22:29, 136kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:13<58:52, 190kB/s]  .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:13<41:24, 269kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 196M/862M [01:15<31:57, 348kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:16<22:28, 493kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<18:08, 609kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 200M/862M [01:17<13:50, 798kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:17<09:53, 1.11MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:19<09:30, 1.15MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 204M/862M [01:19<07:47, 1.41MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:19<05:43, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:21<06:33, 1.66MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:21<05:42, 1.91MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:21<04:15, 2.55MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:23<05:32, 1.96MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 212M/862M [01:23<04:58, 2.18MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:23<03:42, 2.92MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:25<05:09, 2.09MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:25<05:47, 1.86MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:25<04:30, 2.38MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:25<03:19, 3.23MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 220M/862M [01:27<06:26, 1.66MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:27<05:36, 1.90MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:27<04:11, 2.54MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:29<05:23, 1.97MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<05:56, 1.79MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:29<04:36, 2.30MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:29<03:22, 3.14MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 228M/862M [01:31<07:33, 1.40MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 229M/862M [01:31<06:23, 1.65MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:31<04:44, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:33<05:44, 1.83MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:33<05:04, 2.06MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:33<03:48, 2.74MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:35<05:07, 2.03MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 237M/862M [01:35<05:42, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:35<04:26, 2.34MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 240M/862M [01:35<03:15, 3.19MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:37<06:37, 1.56MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:37<05:42, 1.81MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:37<04:14, 2.43MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:39<05:22, 1.91MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:39<05:51, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<04:32, 2.26MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:39<03:18, 3.10MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:41<07:57, 1.28MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:41<06:36, 1.55MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:41<04:49, 2.11MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:43<05:46, 1.76MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:43<06:06, 1.66MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<04:41, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:43<03:24, 2.96MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:45<08:07, 1.24MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<06:43, 1.50MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:45<04:54, 2.05MB/s].vector_cache/glove.6B.zip:  30%|███       | 261M/862M [01:46<05:47, 1.73MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<06:05, 1.65MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<04:41, 2.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:47<03:23, 2.93MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:48<08:26, 1.18MB/s].vector_cache/glove.6B.zip:  31%|███       | 266M/862M [01:49<06:54, 1.44MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<05:04, 1.95MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<05:52, 1.68MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:51<06:06, 1.61MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<04:46, 2.06MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:52<04:54, 2.00MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 274M/862M [01:53<04:26, 2.21MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:53<03:18, 2.95MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<04:36, 2.12MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 278M/862M [01:54<05:11, 1.87MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<04:07, 2.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:55<02:59, 3.23MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<08:44, 1.11MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 282M/862M [01:56<07:06, 1.36MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:57<05:12, 1.85MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<05:53, 1.63MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:58<06:04, 1.58MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<04:45, 2.02MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [01:59<03:25, 2.78MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 290M/862M [02:00<1:10:29, 135kB/s].vector_cache/glove.6B.zip:  34%|███▎      | 291M/862M [02:00<50:16, 189kB/s]  .vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:00<35:20, 269kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:02<26:52, 352kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<20:44, 456kB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:02<14:58, 631kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 298M/862M [02:04<11:57, 786kB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:04<09:19, 1.01MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:04<06:44, 1.39MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<06:53, 1.35MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 303M/862M [02:06<05:46, 1.62MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:06<04:15, 2.18MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<05:09, 1.79MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 307M/862M [02:08<04:33, 2.03MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:08<03:24, 2.70MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:10<04:33, 2.01MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 311M/862M [02:10<05:04, 1.81MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:10<04:00, 2.29MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:12<04:16, 2.14MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 315M/862M [02:12<03:55, 2.32MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:12<02:58, 3.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:14<04:10, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:14<04:46, 1.90MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:14<03:43, 2.43MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:14<02:47, 3.23MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 323M/862M [02:16<04:30, 2.00MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:16<04:05, 2.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:16<03:04, 2.90MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:18<04:14, 2.10MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:18<04:46, 1.86MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:18<03:43, 2.39MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:18<02:44, 3.24MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 331M/862M [02:20<05:19, 1.66MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 332M/862M [02:20<04:37, 1.91MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:20<03:24, 2.58MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 335M/862M [02:22<04:27, 1.97MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:22<03:59, 2.20MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:22<03:00, 2.91MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:24<04:09, 2.09MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:24<03:47, 2.29MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:24<02:52, 3.02MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:26<04:02, 2.13MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:26<04:35, 1.88MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:26<03:34, 2.41MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:26<02:35, 3.31MB/s].vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:28<09:39, 888kB/s] .vector_cache/glove.6B.zip:  40%|████      | 348M/862M [02:28<08:30, 1.01MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<06:20, 1.35MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:28<04:33, 1.87MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:30<06:22, 1.33MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:30<05:19, 1.60MB/s].vector_cache/glove.6B.zip:  41%|████      | 354M/862M [02:30<03:54, 2.17MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:32<04:42, 1.79MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 356M/862M [02:32<05:05, 1.66MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<03:55, 2.15MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:32<02:51, 2.94MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 360M/862M [02:34<05:53, 1.42MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:34<04:59, 1.67MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<03:40, 2.27MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:36<04:30, 1.84MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:36<03:59, 2.07MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<02:58, 2.78MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 368M/862M [02:38<04:02, 2.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:38<03:40, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<02:46, 2.96MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:40<03:51, 2.11MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 373M/862M [02:40<03:31, 2.31MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<02:40, 3.04MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:41<03:46, 2.14MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:42<03:27, 2.33MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:42<02:35, 3.11MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:43<03:43, 2.15MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 381M/862M [02:44<04:15, 1.89MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<03:19, 2.41MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<02:24, 3.31MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:45<09:22, 848kB/s] .vector_cache/glove.6B.zip:  45%|████▍     | 385M/862M [02:46<07:23, 1.07MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 387M/862M [02:46<05:19, 1.49MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:47<05:34, 1.42MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:48<05:30, 1.43MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<04:11, 1.88MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<03:01, 2.59MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:50<06:31, 1.20MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 393M/862M [02:50<06:09, 1.27MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:50<04:38, 1.68MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<03:21, 2.32MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:52<05:49, 1.33MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<04:51, 1.59MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<03:35, 2.15MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:54<04:18, 1.78MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 401M/862M [02:54<04:35, 1.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:54<03:32, 2.16MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:54<02:34, 2.97MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:55<06:04, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:56<06:40, 1.14MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:56<05:04, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:56<03:36, 2.09MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 410M/862M [02:57<16:13, 465kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:58<12:57, 582kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:58<09:23, 802kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<06:38, 1.13MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [02:59<08:48, 849kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 414M/862M [03:00<07:44, 965kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:00<05:47, 1.29MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:00<04:07, 1.79MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:01<23:03, 321kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 418M/862M [03:01<17:42, 418kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:02<12:42, 582kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<08:55, 823kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:03<11:44, 625kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [03:03<09:46, 751kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:04<07:12, 1.02MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<05:06, 1.42MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:05<23:27, 310kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:05<17:57, 405kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:06<12:55, 561kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:06<09:04, 795kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:07<23:44, 303kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:07<18:08, 397kB/s].vector_cache/glove.6B.zip:  50%|████▉     | 431M/862M [03:07<12:59, 553kB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<09:08, 782kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:09<09:54, 721kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:09<08:26, 845kB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:09<06:15, 1.14MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:10<04:26, 1.59MB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:11<23:17, 303kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:11<17:52, 395kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:11<12:51, 548kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:12<09:01, 775kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 442M/862M [03:13<26:13, 267kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:13<19:49, 353kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:13<14:09, 493kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:13<09:58, 697kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:15<09:46, 709kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:15<08:22, 827kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:15<06:14, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:15<04:25, 1.55MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:17<09:57, 689kB/s] .vector_cache/glove.6B.zip:  52%|█████▏    | 451M/862M [03:17<08:21, 820kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:17<06:12, 1.10MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:17<04:23, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<22:28, 302kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:19<17:06, 397kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<12:15, 553kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:19<08:37, 782kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<09:33, 704kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:21<08:02, 835kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<05:58, 1.12MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:21<04:14, 1.57MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<21:59, 302kB/s] .vector_cache/glove.6B.zip:  54%|█████▎    | 463M/862M [03:23<16:44, 397kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<11:59, 553kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:23<08:26, 782kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<09:38, 683kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:25<08:04, 814kB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:25<05:55, 1.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:25<04:13, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:27<06:06, 1.07MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 471M/862M [03:27<05:36, 1.16MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:27<04:11, 1.55MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:27<03:03, 2.12MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:29<04:00, 1.61MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<04:10, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<03:11, 2.01MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:29<02:19, 2.75MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:31<04:06, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:31<04:13, 1.51MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:31<03:14, 1.97MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<02:21, 2.69MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:33<04:03, 1.55MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:33<04:07, 1.53MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:33<03:12, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 487M/862M [03:33<02:17, 2.72MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:35<10:31, 593kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 488M/862M [03:35<08:38, 722kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:35<06:17, 989kB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<04:28, 1.39MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:37<06:13, 990kB/s] .vector_cache/glove.6B.zip:  57%|█████▋    | 492M/862M [03:37<05:49, 1.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:37<04:23, 1.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 496M/862M [03:37<03:08, 1.95MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:39<11:07, 548kB/s] .vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:39<08:58, 679kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:39<06:34, 926kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<04:38, 1.30MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:41<1:29:32, 67.4kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:41<1:03:49, 94.5kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:41<44:51, 134kB/s]   .vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:41<31:13, 191kB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:43<1:30:15, 66.1kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 504M/862M [03:43<1:04:19, 92.7kB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:43<45:11, 132kB/s]   .vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:45<32:20, 182kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:45<23:41, 249kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:45<16:47, 350kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:45<11:43, 498kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:47<14:43, 396kB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 513M/862M [03:47<11:20, 513kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<08:11, 710kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:49<06:39, 866kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:49<07:12, 798kB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 517M/862M [03:49<05:40, 1.01MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:49<04:07, 1.39MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:50<04:03, 1.40MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:51<04:02, 1.41MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:51<03:05, 1.83MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:51<02:13, 2.54MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:52<04:49, 1.16MB/s].vector_cache/glove.6B.zip:  61%|██████    | 525M/862M [03:53<04:10, 1.34MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:53<03:09, 1.77MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:53<02:16, 2.45MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:54<05:25, 1.02MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 529M/862M [03:55<05:01, 1.10MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:55<03:47, 1.46MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:55<02:42, 2.03MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:56<04:23, 1.25MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:57<03:57, 1.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:57<02:56, 1.85MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:57<02:07, 2.56MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:58<05:39, 956kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:59<05:06, 1.06MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:59<03:48, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 540M/862M [03:59<02:43, 1.97MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [04:00<04:11, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:01<03:45, 1.42MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:01<02:49, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<02:54, 1.81MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:02<03:06, 1.70MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:03<02:24, 2.19MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [04:03<01:43, 3.02MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<07:00, 743kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:04<06:00, 866kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:05<04:27, 1.16MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<03:09, 1.63MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<38:57, 132kB/s] .vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:06<28:17, 182kB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:07<20:01, 256kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<14:41, 345kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 558M/862M [04:08<11:18, 448kB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:08<08:09, 619kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:09<05:43, 875kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<39:42, 126kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:10<28:47, 174kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<20:21, 245kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<14:10, 348kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<45:08, 109kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:12<32:34, 151kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<22:57, 214kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:12<16:02, 305kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 570M/862M [04:14<13:06, 371kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:14<09:49, 495kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:14<06:59, 692kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:16<05:46, 831kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<04:54, 977kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<03:38, 1.31MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<03:19, 1.42MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<03:10, 1.49MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<02:23, 1.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:18<01:43, 2.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<04:05, 1.14MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<03:41, 1.26MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<02:47, 1.66MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:22<02:42, 1.70MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:22<02:43, 1.68MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<02:04, 2.20MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:22<01:29, 3.03MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<05:10, 875kB/s] .vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:24<04:28, 1.01MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<03:19, 1.35MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:26<03:03, 1.46MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 595M/862M [04:26<02:58, 1.50MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<02:16, 1.95MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:28<02:18, 1.90MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 599M/862M [04:28<02:26, 1.80MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<01:54, 2.30MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<02:02, 2.11MB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:30<02:14, 1.93MB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<01:43, 2.49MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:30<01:15, 3.40MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:32<03:30, 1.21MB/s].vector_cache/glove.6B.zip:  70%|███████   | 608M/862M [04:32<03:14, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:32<02:27, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<02:24, 1.74MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<02:25, 1.72MB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:34<01:53, 2.20MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<01:59, 2.06MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<02:08, 1.92MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<01:40, 2.43MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:38<01:50, 2.19MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:38<02:01, 2.00MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:38<01:35, 2.53MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:40<01:46, 2.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 624M/862M [04:40<01:57, 2.03MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:40<01:32, 2.56MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<01:43, 2.26MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:42<01:54, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:42<01:30, 2.57MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:44<01:41, 2.27MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:44<01:52, 2.04MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<01:27, 2.63MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:44<01:04, 3.54MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:46<02:19, 1.62MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:46<02:18, 1.63MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:46<01:46, 2.11MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:48<01:51, 2.00MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<01:57, 1.88MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<01:30, 2.44MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:48<01:05, 3.34MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:50<04:13, 858kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<03:38, 996kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:50<02:40, 1.35MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:50<01:54, 1.88MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:51<03:37, 982kB/s] .vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:52<03:12, 1.11MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:52<02:21, 1.50MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:52<01:41, 2.08MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:53<03:30, 996kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<03:06, 1.12MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:54<02:19, 1.49MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:55<02:10, 1.57MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:56<02:09, 1.58MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:56<01:38, 2.08MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:56<01:10, 2.86MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:57<03:00, 1.11MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:58<02:42, 1.24MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<02:02, 1.63MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:59<01:57, 1.68MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:59<01:57, 1.67MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<01:31, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:01<01:35, 2.03MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [05:01<01:41, 1.90MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:02<01:19, 2.41MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [05:03<01:26, 2.18MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<01:34, 1.99MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:04<01:14, 2.52MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:05<01:22, 2.24MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<01:30, 2.03MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:05<01:11, 2.56MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<01:19, 2.26MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<01:28, 2.04MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<01:08, 2.62MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 685M/862M [05:08<00:49, 3.57MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<02:32, 1.16MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<02:18, 1.27MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:09<01:44, 1.68MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:11<01:40, 1.71MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:11<01:41, 1.69MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<01:17, 2.21MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:11<00:55, 3.04MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:13<02:42, 1.04MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:13<02:23, 1.17MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:46, 1.57MB/s].vector_cache/glove.6B.zip:  81%|████████  | 697M/862M [05:13<01:15, 2.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:15<02:48, 973kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:15<02:29, 1.10MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:50, 1.48MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<01:18, 2.06MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:17<02:18, 1.15MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:17<02:06, 1.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:34, 1.68MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<01:07, 2.32MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:19<02:04, 1.25MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:56, 1.33MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:27, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:21<01:25, 1.77MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:21<01:28, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:21<01:08, 2.20MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<01:11, 2.06MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<01:16, 1.92MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<01:00, 2.43MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:25<01:05, 2.19MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:25<01:11, 2.00MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:25<00:56, 2.52MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<01:02, 2.24MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<01:08, 2.03MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<00:54, 2.56MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:29<00:59, 2.26MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:29<01:06, 2.04MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<00:52, 2.57MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<00:57, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:31<01:04, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<00:50, 2.57MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<00:55, 2.27MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:33<01:02, 2.04MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<00:48, 2.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<00:54, 2.27MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:35<01:00, 2.04MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:35<00:46, 2.63MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:35<00:33, 3.57MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:37<01:37, 1.22MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 744M/862M [05:37<01:29, 1.32MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<01:06, 1.76MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:48, 2.41MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:39<01:17, 1.47MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:39<01:15, 1.52MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:39<00:57, 1.98MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:41<00:57, 1.92MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:41<01:01, 1.81MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 753M/862M [05:41<00:46, 2.35MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 755M/862M [05:41<00:33, 3.21MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:43<01:31, 1.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:43<01:24, 1.26MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:43<01:03, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<01:00, 1.70MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<01:01, 1.67MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:45<00:47, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<00:48, 2.03MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<00:52, 1.87MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:47<00:40, 2.39MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:48<00:43, 2.17MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:49<00:47, 1.98MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:49<00:36, 2.54MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:50<00:40, 2.25MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:51<00:44, 2.03MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:51<00:34, 2.60MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<00:24, 3.55MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:52<01:13, 1.17MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:53<01:06, 1.28MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:53<00:49, 1.71MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:35, 2.35MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:54<00:56, 1.46MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:54, 1.51MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:55<00:41, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:40, 1.90MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:42, 1.82MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:57<00:32, 2.33MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:58<00:34, 2.13MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<00:37, 1.96MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:59<00:28, 2.52MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:59<00:20, 3.45MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<01:16, 909kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<01:05, 1.05MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:01<00:48, 1.42MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:01<00:33, 1.98MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<01:45, 619kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<01:25, 763kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<01:01, 1.05MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:03<00:41, 1.47MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<01:26, 705kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<01:11, 852kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:04<00:51, 1.16MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 804M/862M [06:05<00:35, 1.62MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:50, 1.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:06<00:45, 1.24MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:06<00:34, 1.64MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:08<00:31, 1.68MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<00:31, 1.65MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<00:23, 2.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:08<00:16, 2.96MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:10<00:37, 1.29MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<00:35, 1.37MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:10<00:26, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<00:24, 1.79MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<00:25, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:12<00:19, 2.22MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<00:19, 2.07MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<00:20, 1.92MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:14<00:16, 2.43MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<00:16, 2.20MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<00:18, 2.00MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:16<00:13, 2.52MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<00:14, 2.24MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<00:15, 2.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:18<00:12, 2.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<00:12, 2.26MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:20<00:13, 2.03MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:20<00:10, 2.60MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:20<00:06, 3.56MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:22<00:25, 936kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:22<00:22, 1.08MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:15, 1.44MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:24<00:13, 1.53MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:12, 1.56MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:09, 2.03MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:26<00:08, 1.95MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<00:08, 1.84MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:06, 2.35MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:05, 2.14MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:05, 1.96MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:28<00:04, 2.48MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:03, 2.22MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:03, 2.02MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:30<00:02, 2.55MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:32<00:01, 2.26MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:32<00:01, 2.01MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:00, 2.58MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 3.52MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.20MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<21:07:17,  5.26it/s]  0%|          | 636/400000 [00:00<14:46:00,  7.51it/s]  0%|          | 1297/400000 [00:00<10:19:28, 10.73it/s]  0%|          | 1936/400000 [00:00<7:13:15, 15.31it/s]   1%|          | 2591/400000 [00:00<5:03:04, 21.85it/s]  1%|          | 3268/400000 [00:00<3:32:05, 31.18it/s]  1%|          | 3930/400000 [00:00<2:28:30, 44.45it/s]  1%|          | 4584/400000 [00:00<1:44:05, 63.31it/s]  1%|▏         | 5257/400000 [00:00<1:13:01, 90.08it/s]  1%|▏         | 5928/400000 [00:01<51:19, 127.95it/s]   2%|▏         | 6567/400000 [00:01<36:13, 181.00it/s]  2%|▏         | 7234/400000 [00:01<25:36, 255.60it/s]  2%|▏         | 7866/400000 [00:01<18:12, 358.92it/s]  2%|▏         | 8550/400000 [00:01<13:00, 501.45it/s]  2%|▏         | 9268/400000 [00:01<09:21, 695.52it/s]  2%|▏         | 9986/400000 [00:01<06:48, 953.97it/s]  3%|▎         | 10703/400000 [00:01<05:01, 1289.26it/s]  3%|▎         | 11412/400000 [00:01<03:47, 1708.51it/s]  3%|▎         | 12110/400000 [00:02<02:56, 2201.78it/s]  3%|▎         | 12801/400000 [00:02<02:20, 2756.50it/s]  3%|▎         | 13513/400000 [00:02<01:54, 3376.97it/s]  4%|▎         | 14240/400000 [00:02<01:35, 4022.79it/s]  4%|▎         | 14943/400000 [00:02<01:23, 4590.55it/s]  4%|▍         | 15641/400000 [00:02<01:16, 5056.95it/s]  4%|▍         | 16328/400000 [00:02<01:10, 5442.80it/s]  4%|▍         | 17057/400000 [00:02<01:05, 5889.69it/s]  4%|▍         | 17765/400000 [00:02<01:01, 6201.90it/s]  5%|▍         | 18464/400000 [00:02<00:59, 6363.14it/s]  5%|▍         | 19200/400000 [00:03<00:57, 6631.00it/s]  5%|▍         | 19905/400000 [00:03<00:56, 6732.76it/s]  5%|▌         | 20608/400000 [00:03<00:56, 6732.88it/s]  5%|▌         | 21339/400000 [00:03<00:54, 6894.79it/s]  6%|▌         | 22044/400000 [00:03<00:55, 6862.90it/s]  6%|▌         | 22774/400000 [00:03<00:53, 6987.87it/s]  6%|▌         | 23505/400000 [00:03<00:53, 7078.93it/s]  6%|▌         | 24227/400000 [00:03<00:52, 7119.49it/s]  6%|▌         | 24944/400000 [00:03<00:54, 6919.26it/s]  6%|▋         | 25666/400000 [00:03<00:53, 7004.62it/s]  7%|▋         | 26391/400000 [00:04<00:52, 7073.95it/s]  7%|▋         | 27111/400000 [00:04<00:52, 7109.24it/s]  7%|▋         | 27838/400000 [00:04<00:52, 7155.85it/s]  7%|▋         | 28564/400000 [00:04<00:51, 7185.32it/s]  7%|▋         | 29284/400000 [00:04<00:52, 7110.33it/s]  8%|▊         | 30012/400000 [00:04<00:51, 7158.64it/s]  8%|▊         | 30749/400000 [00:04<00:51, 7220.36it/s]  8%|▊         | 31479/400000 [00:04<00:50, 7242.40it/s]  8%|▊         | 32204/400000 [00:04<00:50, 7223.84it/s]  8%|▊         | 32927/400000 [00:04<00:51, 7171.27it/s]  8%|▊         | 33645/400000 [00:05<00:52, 7026.87it/s]  9%|▊         | 34349/400000 [00:05<00:54, 6744.73it/s]  9%|▉         | 35027/400000 [00:05<00:55, 6631.41it/s]  9%|▉         | 35693/400000 [00:05<00:55, 6515.40it/s]  9%|▉         | 36347/400000 [00:05<00:55, 6519.27it/s]  9%|▉         | 37039/400000 [00:05<00:54, 6632.20it/s]  9%|▉         | 37704/400000 [00:05<00:58, 6195.92it/s] 10%|▉         | 38340/400000 [00:05<00:57, 6241.76it/s] 10%|▉         | 39047/400000 [00:05<00:55, 6468.55it/s] 10%|▉         | 39767/400000 [00:06<00:54, 6669.51it/s] 10%|█         | 40484/400000 [00:06<00:53, 6777.50it/s] 10%|█         | 41185/400000 [00:06<00:52, 6844.36it/s] 10%|█         | 41873/400000 [00:06<00:54, 6606.18it/s] 11%|█         | 42538/400000 [00:06<00:54, 6509.34it/s] 11%|█         | 43193/400000 [00:06<00:55, 6377.82it/s] 11%|█         | 43834/400000 [00:06<00:56, 6294.75it/s] 11%|█         | 44469/400000 [00:06<00:56, 6310.96it/s] 11%|█▏        | 45102/400000 [00:06<00:57, 6196.24it/s] 11%|█▏        | 45744/400000 [00:06<00:56, 6261.25it/s] 12%|█▏        | 46444/400000 [00:07<00:54, 6464.40it/s] 12%|█▏        | 47093/400000 [00:07<00:54, 6440.38it/s] 12%|█▏        | 47739/400000 [00:07<00:54, 6422.85it/s] 12%|█▏        | 48383/400000 [00:07<00:54, 6394.34it/s] 12%|█▏        | 49024/400000 [00:07<00:54, 6383.30it/s] 12%|█▏        | 49736/400000 [00:07<00:53, 6587.43it/s] 13%|█▎        | 50426/400000 [00:07<00:52, 6676.26it/s] 13%|█▎        | 51100/400000 [00:07<00:52, 6694.74it/s] 13%|█▎        | 51781/400000 [00:07<00:51, 6726.87it/s] 13%|█▎        | 52455/400000 [00:07<00:52, 6568.12it/s] 13%|█▎        | 53121/400000 [00:08<00:52, 6594.93it/s] 13%|█▎        | 53818/400000 [00:08<00:51, 6701.42it/s] 14%|█▎        | 54505/400000 [00:08<00:51, 6749.16it/s] 14%|█▍        | 55198/400000 [00:08<00:50, 6799.86it/s] 14%|█▍        | 55879/400000 [00:08<00:51, 6718.00it/s] 14%|█▍        | 56552/400000 [00:08<00:51, 6669.79it/s] 14%|█▍        | 57220/400000 [00:08<00:51, 6595.34it/s] 14%|█▍        | 57881/400000 [00:08<00:52, 6529.67it/s] 15%|█▍        | 58535/400000 [00:08<00:52, 6509.72it/s] 15%|█▍        | 59187/400000 [00:08<00:53, 6425.00it/s] 15%|█▍        | 59831/400000 [00:09<00:54, 6269.36it/s] 15%|█▌        | 60487/400000 [00:09<00:53, 6351.49it/s] 15%|█▌        | 61203/400000 [00:09<00:51, 6573.13it/s] 15%|█▌        | 61902/400000 [00:09<00:50, 6689.57it/s] 16%|█▌        | 62574/400000 [00:09<00:51, 6492.51it/s] 16%|█▌        | 63246/400000 [00:09<00:51, 6557.14it/s] 16%|█▌        | 63918/400000 [00:09<00:50, 6604.85it/s] 16%|█▌        | 64626/400000 [00:09<00:49, 6738.39it/s] 16%|█▋        | 65306/400000 [00:09<00:49, 6754.75it/s] 16%|█▋        | 65989/400000 [00:10<00:49, 6774.99it/s] 17%|█▋        | 66719/400000 [00:10<00:48, 6919.59it/s] 17%|█▋        | 67429/400000 [00:10<00:47, 6972.18it/s] 17%|█▋        | 68128/400000 [00:10<00:47, 6945.49it/s] 17%|█▋        | 68824/400000 [00:10<00:48, 6860.67it/s] 17%|█▋        | 69511/400000 [00:10<00:48, 6797.69it/s] 18%|█▊        | 70231/400000 [00:10<00:47, 6912.68it/s] 18%|█▊        | 70936/400000 [00:10<00:47, 6948.10it/s] 18%|█▊        | 71658/400000 [00:10<00:46, 7025.16it/s] 18%|█▊        | 72378/400000 [00:10<00:46, 7075.16it/s] 18%|█▊        | 73087/400000 [00:11<00:46, 7026.65it/s] 18%|█▊        | 73791/400000 [00:11<00:46, 6941.54it/s] 19%|█▊        | 74486/400000 [00:11<00:47, 6818.27it/s] 19%|█▉        | 75188/400000 [00:11<00:47, 6877.09it/s] 19%|█▉        | 75894/400000 [00:11<00:46, 6930.29it/s] 19%|█▉        | 76617/400000 [00:11<00:46, 7016.85it/s] 19%|█▉        | 77320/400000 [00:11<00:46, 6964.20it/s] 20%|█▉        | 78017/400000 [00:11<00:46, 6929.92it/s] 20%|█▉        | 78714/400000 [00:11<00:46, 6938.12it/s] 20%|█▉        | 79409/400000 [00:11<00:47, 6711.82it/s] 20%|██        | 80083/400000 [00:12<00:48, 6532.73it/s] 20%|██        | 80739/400000 [00:12<00:49, 6423.97it/s] 20%|██        | 81384/400000 [00:12<00:50, 6327.65it/s] 21%|██        | 82019/400000 [00:12<00:50, 6256.75it/s] 21%|██        | 82747/400000 [00:12<00:48, 6530.09it/s] 21%|██        | 83479/400000 [00:12<00:46, 6746.97it/s] 21%|██        | 84159/400000 [00:12<00:46, 6732.32it/s] 21%|██        | 84883/400000 [00:12<00:45, 6876.51it/s] 21%|██▏       | 85587/400000 [00:12<00:45, 6919.50it/s] 22%|██▏       | 86282/400000 [00:12<00:45, 6915.77it/s] 22%|██▏       | 86976/400000 [00:13<00:46, 6744.65it/s] 22%|██▏       | 87653/400000 [00:13<00:46, 6713.10it/s] 22%|██▏       | 88368/400000 [00:13<00:45, 6836.75it/s] 22%|██▏       | 89063/400000 [00:13<00:45, 6865.92it/s] 22%|██▏       | 89771/400000 [00:13<00:44, 6926.57it/s] 23%|██▎       | 90465/400000 [00:13<00:44, 6883.02it/s] 23%|██▎       | 91166/400000 [00:13<00:44, 6900.82it/s] 23%|██▎       | 91859/400000 [00:13<00:44, 6908.77it/s] 23%|██▎       | 92592/400000 [00:13<00:43, 7021.33it/s] 23%|██▎       | 93295/400000 [00:13<00:43, 7009.60it/s] 23%|██▎       | 93997/400000 [00:14<00:44, 6944.50it/s] 24%|██▎       | 94692/400000 [00:14<00:44, 6886.55it/s] 24%|██▍       | 95411/400000 [00:14<00:43, 6973.19it/s] 24%|██▍       | 96139/400000 [00:14<00:43, 7061.23it/s] 24%|██▍       | 96864/400000 [00:14<00:42, 7114.46it/s] 24%|██▍       | 97598/400000 [00:14<00:42, 7178.86it/s] 25%|██▍       | 98317/400000 [00:14<00:43, 6978.92it/s] 25%|██▍       | 99017/400000 [00:14<00:46, 6534.27it/s] 25%|██▍       | 99678/400000 [00:14<00:46, 6446.19it/s] 25%|██▌       | 100343/400000 [00:15<00:46, 6504.03it/s] 25%|██▌       | 100998/400000 [00:15<00:46, 6444.21it/s] 25%|██▌       | 101697/400000 [00:15<00:45, 6598.54it/s] 26%|██▌       | 102424/400000 [00:15<00:43, 6784.72it/s] 26%|██▌       | 103106/400000 [00:15<00:44, 6746.65it/s] 26%|██▌       | 103784/400000 [00:15<00:45, 6578.72it/s] 26%|██▌       | 104445/400000 [00:15<00:45, 6484.95it/s] 26%|██▋       | 105096/400000 [00:15<00:46, 6284.58it/s] 26%|██▋       | 105793/400000 [00:15<00:45, 6475.46it/s] 27%|██▋       | 106512/400000 [00:15<00:43, 6673.47it/s] 27%|██▋       | 107239/400000 [00:16<00:42, 6840.32it/s] 27%|██▋       | 107927/400000 [00:16<00:42, 6820.27it/s] 27%|██▋       | 108623/400000 [00:16<00:42, 6861.56it/s] 27%|██▋       | 109312/400000 [00:16<00:42, 6827.62it/s] 27%|██▋       | 109997/400000 [00:16<00:42, 6813.86it/s] 28%|██▊       | 110708/400000 [00:16<00:41, 6898.36it/s] 28%|██▊       | 111415/400000 [00:16<00:41, 6947.27it/s] 28%|██▊       | 112111/400000 [00:16<00:41, 6888.66it/s] 28%|██▊       | 112801/400000 [00:16<00:41, 6879.07it/s] 28%|██▊       | 113525/400000 [00:16<00:41, 6982.09it/s] 29%|██▊       | 114224/400000 [00:17<00:41, 6889.37it/s] 29%|██▊       | 114914/400000 [00:17<00:41, 6868.59it/s] 29%|██▉       | 115602/400000 [00:17<00:41, 6862.81it/s] 29%|██▉       | 116289/400000 [00:17<00:42, 6666.35it/s] 29%|██▉       | 116958/400000 [00:17<00:44, 6417.26it/s] 29%|██▉       | 117660/400000 [00:17<00:42, 6584.39it/s] 30%|██▉       | 118372/400000 [00:17<00:41, 6734.61it/s] 30%|██▉       | 119056/400000 [00:17<00:41, 6765.75it/s] 30%|██▉       | 119758/400000 [00:17<00:40, 6838.80it/s] 30%|███       | 120444/400000 [00:18<00:41, 6797.18it/s] 30%|███       | 121125/400000 [00:18<00:41, 6723.55it/s] 30%|███       | 121799/400000 [00:18<00:42, 6607.07it/s] 31%|███       | 122461/400000 [00:18<00:42, 6595.68it/s] 31%|███       | 123124/400000 [00:18<00:41, 6605.61it/s] 31%|███       | 123793/400000 [00:18<00:41, 6630.52it/s] 31%|███       | 124464/400000 [00:18<00:41, 6652.76it/s] 31%|███▏      | 125189/400000 [00:18<00:40, 6820.00it/s] 31%|███▏      | 125917/400000 [00:18<00:39, 6951.53it/s] 32%|███▏      | 126650/400000 [00:18<00:38, 7058.86it/s] 32%|███▏      | 127358/400000 [00:19<00:40, 6735.95it/s] 32%|███▏      | 128036/400000 [00:19<00:41, 6491.71it/s] 32%|███▏      | 128749/400000 [00:19<00:40, 6666.97it/s] 32%|███▏      | 129428/400000 [00:19<00:40, 6701.06it/s] 33%|███▎      | 130132/400000 [00:19<00:39, 6797.10it/s] 33%|███▎      | 130839/400000 [00:19<00:39, 6875.50it/s] 33%|███▎      | 131551/400000 [00:19<00:38, 6946.55it/s] 33%|███▎      | 132248/400000 [00:19<00:38, 6936.36it/s] 33%|███▎      | 132967/400000 [00:19<00:38, 7008.29it/s] 33%|███▎      | 133669/400000 [00:19<00:38, 6948.79it/s] 34%|███▎      | 134373/400000 [00:20<00:38, 6975.84it/s] 34%|███▍      | 135072/400000 [00:20<00:38, 6917.76it/s] 34%|███▍      | 135781/400000 [00:20<00:37, 6967.09it/s] 34%|███▍      | 136479/400000 [00:20<00:37, 6969.60it/s] 34%|███▍      | 137179/400000 [00:20<00:37, 6977.65it/s] 34%|███▍      | 137877/400000 [00:20<00:38, 6791.73it/s] 35%|███▍      | 138587/400000 [00:20<00:38, 6879.03it/s] 35%|███▍      | 139306/400000 [00:20<00:37, 6966.62it/s] 35%|███▌      | 140004/400000 [00:20<00:37, 6948.14it/s] 35%|███▌      | 140700/400000 [00:20<00:38, 6806.91it/s] 35%|███▌      | 141382/400000 [00:21<00:38, 6735.99it/s] 36%|███▌      | 142057/400000 [00:21<00:39, 6477.08it/s] 36%|███▌      | 142772/400000 [00:21<00:38, 6665.25it/s] 36%|███▌      | 143483/400000 [00:21<00:37, 6792.18it/s] 36%|███▌      | 144166/400000 [00:21<00:37, 6779.65it/s] 36%|███▌      | 144870/400000 [00:21<00:37, 6843.40it/s] 36%|███▋      | 145563/400000 [00:21<00:37, 6867.40it/s] 37%|███▋      | 146251/400000 [00:21<00:37, 6724.66it/s] 37%|███▋      | 146967/400000 [00:21<00:36, 6848.12it/s] 37%|███▋      | 147691/400000 [00:22<00:36, 6958.62it/s] 37%|███▋      | 148389/400000 [00:22<00:37, 6645.13it/s] 37%|███▋      | 149058/400000 [00:22<00:38, 6501.87it/s] 37%|███▋      | 149779/400000 [00:22<00:37, 6697.88it/s] 38%|███▊      | 150453/400000 [00:22<00:37, 6688.54it/s] 38%|███▊      | 151153/400000 [00:22<00:36, 6778.99it/s] 38%|███▊      | 151834/400000 [00:22<00:36, 6707.44it/s] 38%|███▊      | 152507/400000 [00:22<00:37, 6619.11it/s] 38%|███▊      | 153207/400000 [00:22<00:36, 6727.68it/s] 38%|███▊      | 153933/400000 [00:22<00:35, 6877.72it/s] 39%|███▊      | 154623/400000 [00:23<00:37, 6603.12it/s] 39%|███▉      | 155287/400000 [00:23<00:37, 6450.66it/s] 39%|███▉      | 156007/400000 [00:23<00:36, 6656.58it/s] 39%|███▉      | 156691/400000 [00:23<00:36, 6709.54it/s] 39%|███▉      | 157421/400000 [00:23<00:35, 6876.34it/s] 40%|███▉      | 158153/400000 [00:23<00:34, 7001.87it/s] 40%|███▉      | 158856/400000 [00:23<00:34, 6984.41it/s] 40%|███▉      | 159557/400000 [00:23<00:37, 6330.10it/s] 40%|████      | 160204/400000 [00:23<00:37, 6318.83it/s] 40%|████      | 160897/400000 [00:23<00:36, 6489.03it/s] 40%|████      | 161554/400000 [00:24<00:37, 6430.66it/s] 41%|████      | 162252/400000 [00:24<00:36, 6585.91it/s] 41%|████      | 162979/400000 [00:24<00:34, 6775.84it/s] 41%|████      | 163667/400000 [00:24<00:34, 6804.69it/s] 41%|████      | 164384/400000 [00:24<00:34, 6909.35it/s] 41%|████▏     | 165119/400000 [00:24<00:33, 7033.86it/s] 41%|████▏     | 165844/400000 [00:24<00:33, 7094.93it/s] 42%|████▏     | 166579/400000 [00:24<00:32, 7168.03it/s] 42%|████▏     | 167298/400000 [00:24<00:32, 7154.64it/s] 42%|████▏     | 168015/400000 [00:25<00:32, 7110.38it/s] 42%|████▏     | 168727/400000 [00:25<00:32, 7054.59it/s] 42%|████▏     | 169434/400000 [00:25<00:34, 6769.87it/s] 43%|████▎     | 170114/400000 [00:25<00:34, 6670.78it/s] 43%|████▎     | 170784/400000 [00:25<00:34, 6594.67it/s] 43%|████▎     | 171476/400000 [00:25<00:34, 6686.99it/s] 43%|████▎     | 172147/400000 [00:25<00:34, 6555.72it/s] 43%|████▎     | 172855/400000 [00:25<00:33, 6702.05it/s] 43%|████▎     | 173581/400000 [00:25<00:33, 6857.64it/s] 44%|████▎     | 174302/400000 [00:25<00:32, 6959.14it/s] 44%|████▍     | 175033/400000 [00:26<00:31, 7058.65it/s] 44%|████▍     | 175755/400000 [00:26<00:31, 7104.80it/s] 44%|████▍     | 176467/400000 [00:26<00:31, 7030.57it/s] 44%|████▍     | 177199/400000 [00:26<00:31, 7107.52it/s] 44%|████▍     | 177911/400000 [00:26<00:31, 7075.45it/s] 45%|████▍     | 178625/400000 [00:26<00:31, 7093.17it/s] 45%|████▍     | 179340/400000 [00:26<00:31, 7107.47it/s] 45%|████▌     | 180052/400000 [00:26<00:31, 7075.39it/s] 45%|████▌     | 180760/400000 [00:26<00:32, 6759.14it/s] 45%|████▌     | 181440/400000 [00:26<00:33, 6621.69it/s] 46%|████▌     | 182106/400000 [00:27<00:33, 6493.49it/s] 46%|████▌     | 182758/400000 [00:27<00:34, 6385.33it/s] 46%|████▌     | 183460/400000 [00:27<00:32, 6562.00it/s] 46%|████▌     | 184183/400000 [00:27<00:31, 6747.06it/s] 46%|████▌     | 184861/400000 [00:27<00:32, 6697.67it/s] 46%|████▋     | 185588/400000 [00:27<00:31, 6858.98it/s] 47%|████▋     | 186277/400000 [00:27<00:31, 6853.67it/s] 47%|████▋     | 186981/400000 [00:27<00:30, 6907.27it/s] 47%|████▋     | 187674/400000 [00:27<00:30, 6902.69it/s] 47%|████▋     | 188396/400000 [00:27<00:30, 6993.61it/s] 47%|████▋     | 189097/400000 [00:28<00:30, 6939.20it/s] 47%|████▋     | 189813/400000 [00:28<00:30, 7001.09it/s] 48%|████▊     | 190514/400000 [00:28<00:30, 6838.18it/s] 48%|████▊     | 191245/400000 [00:28<00:29, 6970.69it/s] 48%|████▊     | 191944/400000 [00:28<00:31, 6547.10it/s] 48%|████▊     | 192622/400000 [00:28<00:31, 6614.12it/s] 48%|████▊     | 193289/400000 [00:28<00:32, 6324.34it/s] 48%|████▊     | 193942/400000 [00:28<00:32, 6382.21it/s] 49%|████▊     | 194595/400000 [00:28<00:31, 6425.46it/s] 49%|████▉     | 195248/400000 [00:29<00:31, 6454.46it/s] 49%|████▉     | 195896/400000 [00:29<00:32, 6306.18it/s] 49%|████▉     | 196529/400000 [00:29<00:32, 6273.35it/s] 49%|████▉     | 197246/400000 [00:29<00:31, 6516.21it/s] 49%|████▉     | 197917/400000 [00:29<00:30, 6571.10it/s] 50%|████▉     | 198636/400000 [00:29<00:29, 6743.69it/s] 50%|████▉     | 199337/400000 [00:29<00:29, 6820.45it/s] 50%|█████     | 200057/400000 [00:29<00:28, 6930.03it/s] 50%|█████     | 200753/400000 [00:29<00:29, 6801.71it/s] 50%|█████     | 201436/400000 [00:29<00:29, 6697.37it/s] 51%|█████     | 202108/400000 [00:30<00:30, 6499.82it/s] 51%|█████     | 202804/400000 [00:30<00:29, 6630.20it/s] 51%|█████     | 203470/400000 [00:30<00:31, 6249.71it/s] 51%|█████     | 204164/400000 [00:30<00:30, 6440.17it/s] 51%|█████     | 204900/400000 [00:30<00:29, 6689.56it/s] 51%|█████▏    | 205625/400000 [00:30<00:28, 6846.79it/s] 52%|█████▏    | 206316/400000 [00:30<00:28, 6820.11it/s] 52%|█████▏    | 207012/400000 [00:30<00:28, 6860.28it/s] 52%|█████▏    | 207724/400000 [00:30<00:27, 6935.70it/s] 52%|█████▏    | 208455/400000 [00:30<00:27, 7043.89it/s] 52%|█████▏    | 209178/400000 [00:31<00:26, 7096.09it/s] 52%|█████▏    | 209902/400000 [00:31<00:26, 7138.10it/s] 53%|█████▎    | 210617/400000 [00:31<00:26, 7140.65it/s] 53%|█████▎    | 211335/400000 [00:31<00:26, 7149.95it/s] 53%|█████▎    | 212053/400000 [00:31<00:26, 7156.06it/s] 53%|█████▎    | 212769/400000 [00:31<00:26, 7096.83it/s] 53%|█████▎    | 213484/400000 [00:31<00:26, 7109.67it/s] 54%|█████▎    | 214196/400000 [00:31<00:26, 6975.39it/s] 54%|█████▎    | 214897/400000 [00:31<00:26, 6983.64it/s] 54%|█████▍    | 215621/400000 [00:32<00:26, 7056.21it/s] 54%|█████▍    | 216328/400000 [00:32<00:26, 7038.84it/s] 54%|█████▍    | 217033/400000 [00:32<00:26, 7001.61it/s] 54%|█████▍    | 217734/400000 [00:32<00:26, 6895.34it/s] 55%|█████▍    | 218425/400000 [00:32<00:26, 6886.87it/s] 55%|█████▍    | 219115/400000 [00:32<00:26, 6798.18it/s] 55%|█████▍    | 219796/400000 [00:32<00:27, 6590.21it/s] 55%|█████▌    | 220457/400000 [00:32<00:28, 6235.91it/s] 55%|█████▌    | 221096/400000 [00:32<00:28, 6278.47it/s] 55%|█████▌    | 221728/400000 [00:32<00:28, 6266.63it/s] 56%|█████▌    | 222418/400000 [00:33<00:27, 6443.21it/s] 56%|█████▌    | 223077/400000 [00:33<00:27, 6484.77it/s] 56%|█████▌    | 223730/400000 [00:33<00:27, 6496.84it/s] 56%|█████▌    | 224443/400000 [00:33<00:26, 6673.37it/s] 56%|█████▋    | 225130/400000 [00:33<00:25, 6730.72it/s] 56%|█████▋    | 225842/400000 [00:33<00:25, 6842.27it/s] 57%|█████▋    | 226561/400000 [00:33<00:24, 6938.37it/s] 57%|█████▋    | 227257/400000 [00:33<00:25, 6684.81it/s] 57%|█████▋    | 227951/400000 [00:33<00:25, 6756.50it/s] 57%|█████▋    | 228665/400000 [00:33<00:24, 6866.35it/s] 57%|█████▋    | 229384/400000 [00:34<00:24, 6958.55it/s] 58%|█████▊    | 230107/400000 [00:34<00:24, 7034.99it/s] 58%|█████▊    | 230830/400000 [00:34<00:23, 7089.90it/s] 58%|█████▊    | 231541/400000 [00:34<00:23, 7095.30it/s] 58%|█████▊    | 232252/400000 [00:34<00:23, 7029.58it/s] 58%|█████▊    | 232961/400000 [00:34<00:23, 7046.92it/s] 58%|█████▊    | 233667/400000 [00:34<00:23, 7030.64it/s] 59%|█████▊    | 234395/400000 [00:34<00:23, 7102.19it/s] 59%|█████▉    | 235114/400000 [00:34<00:23, 7126.79it/s] 59%|█████▉    | 235827/400000 [00:34<00:23, 7088.47it/s] 59%|█████▉    | 236544/400000 [00:35<00:22, 7112.26it/s] 59%|█████▉    | 237256/400000 [00:35<00:23, 7048.94it/s] 59%|█████▉    | 237985/400000 [00:35<00:22, 7117.87it/s] 60%|█████▉    | 238709/400000 [00:35<00:22, 7152.17it/s] 60%|█████▉    | 239425/400000 [00:35<00:22, 7053.44it/s] 60%|██████    | 240131/400000 [00:35<00:22, 7028.73it/s] 60%|██████    | 240835/400000 [00:35<00:22, 6963.03it/s] 60%|██████    | 241532/400000 [00:35<00:22, 6904.99it/s] 61%|██████    | 242247/400000 [00:35<00:22, 6976.11it/s] 61%|██████    | 242979/400000 [00:35<00:22, 7074.81it/s] 61%|██████    | 243704/400000 [00:36<00:21, 7124.27it/s] 61%|██████    | 244417/400000 [00:36<00:21, 7092.63it/s] 61%|██████▏   | 245137/400000 [00:36<00:21, 7122.44it/s] 61%|██████▏   | 245850/400000 [00:36<00:21, 7031.32it/s] 62%|██████▏   | 246554/400000 [00:36<00:22, 6853.11it/s] 62%|██████▏   | 247241/400000 [00:36<00:23, 6618.99it/s] 62%|██████▏   | 247966/400000 [00:36<00:22, 6795.55it/s] 62%|██████▏   | 248674/400000 [00:36<00:22, 6874.96it/s] 62%|██████▏   | 249366/400000 [00:36<00:21, 6886.99it/s] 63%|██████▎   | 250090/400000 [00:37<00:21, 6987.51it/s] 63%|██████▎   | 250798/400000 [00:37<00:21, 7013.78it/s] 63%|██████▎   | 251532/400000 [00:37<00:20, 7107.69it/s] 63%|██████▎   | 252262/400000 [00:37<00:20, 7161.85it/s] 63%|██████▎   | 252980/400000 [00:37<00:20, 7146.04it/s] 63%|██████▎   | 253696/400000 [00:37<00:20, 7116.19it/s] 64%|██████▎   | 254421/400000 [00:37<00:20, 7155.58it/s] 64%|██████▍   | 255137/400000 [00:37<00:20, 7147.00it/s] 64%|██████▍   | 255852/400000 [00:37<00:20, 7144.93it/s] 64%|██████▍   | 256567/400000 [00:37<00:20, 7086.03it/s] 64%|██████▍   | 257279/400000 [00:38<00:20, 7095.11it/s] 64%|██████▍   | 258000/400000 [00:38<00:19, 7128.01it/s] 65%|██████▍   | 258713/400000 [00:38<00:19, 7065.65it/s] 65%|██████▍   | 259420/400000 [00:38<00:20, 6995.27it/s] 65%|██████▌   | 260141/400000 [00:38<00:19, 7057.40it/s] 65%|██████▌   | 260875/400000 [00:38<00:19, 7137.93it/s] 65%|██████▌   | 261590/400000 [00:38<00:19, 7121.24it/s] 66%|██████▌   | 262305/400000 [00:38<00:19, 7128.47it/s] 66%|██████▌   | 263019/400000 [00:38<00:20, 6765.06it/s] 66%|██████▌   | 263721/400000 [00:38<00:19, 6838.62it/s] 66%|██████▌   | 264414/400000 [00:39<00:19, 6863.53it/s] 66%|██████▋   | 265138/400000 [00:39<00:19, 6969.95it/s] 66%|██████▋   | 265837/400000 [00:39<00:19, 6933.90it/s] 67%|██████▋   | 266532/400000 [00:39<00:19, 6937.15it/s] 67%|██████▋   | 267227/400000 [00:39<00:20, 6623.71it/s] 67%|██████▋   | 267894/400000 [00:39<00:20, 6502.78it/s] 67%|██████▋   | 268586/400000 [00:39<00:19, 6622.10it/s] 67%|██████▋   | 269309/400000 [00:39<00:19, 6792.52it/s] 68%|██████▊   | 270009/400000 [00:39<00:18, 6852.89it/s] 68%|██████▊   | 270697/400000 [00:39<00:19, 6710.41it/s] 68%|██████▊   | 271384/400000 [00:40<00:19, 6754.87it/s] 68%|██████▊   | 272113/400000 [00:40<00:18, 6906.08it/s] 68%|██████▊   | 272838/400000 [00:40<00:18, 7005.42it/s] 68%|██████▊   | 273564/400000 [00:40<00:17, 7077.31it/s] 69%|██████▊   | 274286/400000 [00:40<00:17, 7117.54it/s] 69%|██████▊   | 274999/400000 [00:40<00:18, 6807.21it/s] 69%|██████▉   | 275684/400000 [00:40<00:18, 6729.45it/s] 69%|██████▉   | 276360/400000 [00:40<00:18, 6629.54it/s] 69%|██████▉   | 277026/400000 [00:40<00:18, 6600.42it/s] 69%|██████▉   | 277688/400000 [00:41<00:18, 6564.39it/s] 70%|██████▉   | 278407/400000 [00:41<00:18, 6739.95it/s] 70%|██████▉   | 279083/400000 [00:41<00:19, 6346.37it/s] 70%|██████▉   | 279734/400000 [00:41<00:18, 6392.87it/s] 70%|███████   | 280385/400000 [00:41<00:18, 6427.39it/s] 70%|███████   | 281063/400000 [00:41<00:18, 6528.65it/s] 70%|███████   | 281792/400000 [00:41<00:17, 6737.27it/s] 71%|███████   | 282531/400000 [00:41<00:16, 6918.68it/s] 71%|███████   | 283227/400000 [00:41<00:16, 6924.02it/s] 71%|███████   | 283953/400000 [00:41<00:16, 7020.52it/s] 71%|███████   | 284658/400000 [00:42<00:16, 7009.44it/s] 71%|███████▏  | 285392/400000 [00:42<00:16, 7105.31it/s] 72%|███████▏  | 286128/400000 [00:42<00:15, 7178.16it/s] 72%|███████▏  | 286863/400000 [00:42<00:15, 7228.80it/s] 72%|███████▏  | 287587/400000 [00:42<00:16, 6718.59it/s] 72%|███████▏  | 288281/400000 [00:42<00:16, 6782.68it/s] 72%|███████▏  | 288965/400000 [00:42<00:16, 6792.25it/s] 72%|███████▏  | 289649/400000 [00:42<00:16, 6686.74it/s] 73%|███████▎  | 290321/400000 [00:42<00:16, 6642.24it/s] 73%|███████▎  | 290988/400000 [00:42<00:16, 6569.19it/s] 73%|███████▎  | 291647/400000 [00:43<00:16, 6436.21it/s] 73%|███████▎  | 292353/400000 [00:43<00:16, 6611.30it/s] 73%|███████▎  | 293075/400000 [00:43<00:15, 6782.54it/s] 73%|███████▎  | 293810/400000 [00:43<00:15, 6941.50it/s] 74%|███████▎  | 294546/400000 [00:43<00:14, 7059.68it/s] 74%|███████▍  | 295266/400000 [00:43<00:14, 7101.01it/s] 74%|███████▍  | 295978/400000 [00:43<00:14, 7049.30it/s] 74%|███████▍  | 296703/400000 [00:43<00:14, 7105.74it/s] 74%|███████▍  | 297420/400000 [00:43<00:14, 7123.61it/s] 75%|███████▍  | 298149/400000 [00:43<00:14, 7172.49it/s] 75%|███████▍  | 298867/400000 [00:44<00:14, 7154.60it/s] 75%|███████▍  | 299583/400000 [00:44<00:14, 7083.85it/s] 75%|███████▌  | 300292/400000 [00:44<00:14, 6960.55it/s] 75%|███████▌  | 301012/400000 [00:44<00:14, 7029.49it/s] 75%|███████▌  | 301746/400000 [00:44<00:13, 7118.67it/s] 76%|███████▌  | 302459/400000 [00:44<00:13, 7018.11it/s] 76%|███████▌  | 303162/400000 [00:44<00:14, 6608.14it/s] 76%|███████▌  | 303850/400000 [00:44<00:14, 6685.14it/s] 76%|███████▌  | 304523/400000 [00:44<00:14, 6432.77it/s] 76%|███████▋  | 305172/400000 [00:45<00:14, 6423.19it/s] 76%|███████▋  | 305823/400000 [00:45<00:14, 6448.17it/s] 77%|███████▋  | 306480/400000 [00:45<00:14, 6483.25it/s] 77%|███████▋  | 307140/400000 [00:45<00:14, 6515.59it/s] 77%|███████▋  | 307856/400000 [00:45<00:13, 6695.50it/s] 77%|███████▋  | 308562/400000 [00:45<00:13, 6799.44it/s] 77%|███████▋  | 309254/400000 [00:45<00:13, 6833.29it/s] 77%|███████▋  | 309987/400000 [00:45<00:12, 6972.50it/s] 78%|███████▊  | 310714/400000 [00:45<00:12, 7056.65it/s] 78%|███████▊  | 311442/400000 [00:45<00:12, 7120.33it/s] 78%|███████▊  | 312176/400000 [00:46<00:12, 7182.80it/s] 78%|███████▊  | 312896/400000 [00:46<00:12, 6836.63it/s] 78%|███████▊  | 313584/400000 [00:46<00:12, 6675.34it/s] 79%|███████▊  | 314313/400000 [00:46<00:12, 6847.34it/s] 79%|███████▉  | 315002/400000 [00:46<00:12, 6738.75it/s] 79%|███████▉  | 315679/400000 [00:46<00:12, 6661.59it/s] 79%|███████▉  | 316348/400000 [00:46<00:12, 6600.89it/s] 79%|███████▉  | 317026/400000 [00:46<00:12, 6652.58it/s] 79%|███████▉  | 317744/400000 [00:46<00:12, 6801.28it/s] 80%|███████▉  | 318467/400000 [00:46<00:11, 6922.54it/s] 80%|███████▉  | 319195/400000 [00:47<00:11, 7023.86it/s] 80%|███████▉  | 319908/400000 [00:47<00:11, 7053.15it/s] 80%|████████  | 320615/400000 [00:47<00:11, 6862.66it/s] 80%|████████  | 321304/400000 [00:47<00:11, 6793.00it/s] 81%|████████  | 322012/400000 [00:47<00:11, 6874.42it/s] 81%|████████  | 322701/400000 [00:47<00:11, 6844.38it/s] 81%|████████  | 323387/400000 [00:47<00:11, 6801.14it/s] 81%|████████  | 324068/400000 [00:47<00:11, 6614.05it/s] 81%|████████  | 324759/400000 [00:47<00:11, 6698.11it/s] 81%|████████▏ | 325451/400000 [00:48<00:11, 6761.01it/s] 82%|████████▏ | 326141/400000 [00:48<00:10, 6800.05it/s] 82%|████████▏ | 326826/400000 [00:48<00:10, 6813.53it/s] 82%|████████▏ | 327542/400000 [00:48<00:10, 6913.66it/s] 82%|████████▏ | 328235/400000 [00:48<00:10, 6777.69it/s] 82%|████████▏ | 328914/400000 [00:48<00:10, 6708.18it/s] 82%|████████▏ | 329586/400000 [00:48<00:10, 6653.82it/s] 83%|████████▎ | 330253/400000 [00:48<00:10, 6379.81it/s] 83%|████████▎ | 330934/400000 [00:48<00:10, 6501.89it/s] 83%|████████▎ | 331647/400000 [00:48<00:10, 6677.68it/s] 83%|████████▎ | 332363/400000 [00:49<00:09, 6813.75it/s] 83%|████████▎ | 333071/400000 [00:49<00:09, 6890.21it/s] 83%|████████▎ | 333784/400000 [00:49<00:09, 6959.47it/s] 84%|████████▎ | 334482/400000 [00:49<00:09, 6621.31it/s] 84%|████████▍ | 335149/400000 [00:49<00:09, 6576.25it/s] 84%|████████▍ | 335810/400000 [00:49<00:09, 6548.79it/s] 84%|████████▍ | 336492/400000 [00:49<00:09, 6627.85it/s] 84%|████████▍ | 337197/400000 [00:49<00:09, 6748.80it/s] 84%|████████▍ | 337917/400000 [00:49<00:09, 6875.67it/s] 85%|████████▍ | 338607/400000 [00:49<00:09, 6523.93it/s] 85%|████████▍ | 339327/400000 [00:50<00:09, 6712.61it/s] 85%|████████▌ | 340018/400000 [00:50<00:08, 6769.28it/s] 85%|████████▌ | 340699/400000 [00:50<00:09, 6582.48it/s] 85%|████████▌ | 341420/400000 [00:50<00:08, 6758.18it/s] 86%|████████▌ | 342144/400000 [00:50<00:08, 6892.27it/s] 86%|████████▌ | 342837/400000 [00:50<00:08, 6719.24it/s] 86%|████████▌ | 343549/400000 [00:50<00:08, 6833.26it/s] 86%|████████▌ | 344281/400000 [00:50<00:07, 6970.22it/s] 86%|████████▋ | 345002/400000 [00:50<00:07, 7039.25it/s] 86%|████████▋ | 345708/400000 [00:50<00:07, 6959.11it/s] 87%|████████▋ | 346406/400000 [00:51<00:08, 6679.22it/s] 87%|████████▋ | 347078/400000 [00:51<00:08, 6341.73it/s] 87%|████████▋ | 347770/400000 [00:51<00:08, 6503.83it/s] 87%|████████▋ | 348426/400000 [00:51<00:07, 6475.69it/s] 87%|████████▋ | 349141/400000 [00:51<00:07, 6663.41it/s] 87%|████████▋ | 349866/400000 [00:51<00:07, 6827.77it/s] 88%|████████▊ | 350589/400000 [00:51<00:07, 6941.12it/s] 88%|████████▊ | 351287/400000 [00:51<00:07, 6826.05it/s] 88%|████████▊ | 352015/400000 [00:51<00:06, 6954.44it/s] 88%|████████▊ | 352739/400000 [00:52<00:06, 7037.64it/s] 88%|████████▊ | 353452/400000 [00:52<00:06, 7063.83it/s] 89%|████████▊ | 354160/400000 [00:52<00:06, 6668.29it/s] 89%|████████▊ | 354833/400000 [00:52<00:06, 6545.74it/s] 89%|████████▉ | 355492/400000 [00:52<00:06, 6423.71it/s] 89%|████████▉ | 356166/400000 [00:52<00:06, 6514.21it/s] 89%|████████▉ | 356882/400000 [00:52<00:06, 6695.23it/s] 89%|████████▉ | 357588/400000 [00:52<00:06, 6798.59it/s] 90%|████████▉ | 358304/400000 [00:52<00:06, 6902.99it/s] 90%|████████▉ | 358997/400000 [00:52<00:06, 6706.37it/s] 90%|████████▉ | 359671/400000 [00:53<00:06, 6385.94it/s] 90%|█████████ | 360315/400000 [00:53<00:06, 6359.72it/s] 90%|█████████ | 360994/400000 [00:53<00:06, 6480.99it/s] 90%|█████████ | 361705/400000 [00:53<00:05, 6655.70it/s] 91%|█████████ | 362374/400000 [00:53<00:05, 6558.07it/s] 91%|█████████ | 363033/400000 [00:53<00:05, 6510.61it/s] 91%|█████████ | 363687/400000 [00:53<00:05, 6383.35it/s] 91%|█████████ | 364328/400000 [00:53<00:05, 6162.15it/s] 91%|█████████ | 364963/400000 [00:53<00:05, 6215.78it/s] 91%|█████████▏| 365606/400000 [00:54<00:05, 6278.28it/s] 92%|█████████▏| 366255/400000 [00:54<00:05, 6339.01it/s] 92%|█████████▏| 366891/400000 [00:54<00:05, 6307.78it/s] 92%|█████████▏| 367619/400000 [00:54<00:04, 6570.94it/s] 92%|█████████▏| 368288/400000 [00:54<00:04, 6603.84it/s] 92%|█████████▏| 369008/400000 [00:54<00:04, 6770.26it/s] 92%|█████████▏| 369736/400000 [00:54<00:04, 6913.55it/s] 93%|█████████▎| 370431/400000 [00:54<00:04, 6910.65it/s] 93%|█████████▎| 371163/400000 [00:54<00:04, 7027.08it/s] 93%|█████████▎| 371883/400000 [00:54<00:03, 7077.53it/s] 93%|█████████▎| 372593/400000 [00:55<00:03, 6966.66it/s] 93%|█████████▎| 373316/400000 [00:55<00:03, 7042.08it/s] 94%|█████████▎| 374023/400000 [00:55<00:03, 7047.85it/s] 94%|█████████▎| 374729/400000 [00:55<00:03, 6857.35it/s] 94%|█████████▍| 375417/400000 [00:55<00:03, 6618.06it/s] 94%|█████████▍| 376101/400000 [00:55<00:03, 6682.41it/s] 94%|█████████▍| 376772/400000 [00:55<00:03, 6613.32it/s] 94%|█████████▍| 377489/400000 [00:55<00:03, 6770.85it/s] 95%|█████████▍| 378205/400000 [00:55<00:03, 6881.45it/s] 95%|█████████▍| 378926/400000 [00:55<00:03, 6975.36it/s] 95%|█████████▍| 379644/400000 [00:56<00:02, 7034.32it/s] 95%|█████████▌| 380373/400000 [00:56<00:02, 7108.81it/s] 95%|█████████▌| 381085/400000 [00:56<00:02, 6861.98it/s] 95%|█████████▌| 381811/400000 [00:56<00:02, 6975.27it/s] 96%|█████████▌| 382532/400000 [00:56<00:02, 7041.91it/s] 96%|█████████▌| 383238/400000 [00:56<00:02, 7017.87it/s] 96%|█████████▌| 383942/400000 [00:56<00:02, 7023.46it/s] 96%|█████████▌| 384657/400000 [00:56<00:02, 7059.24it/s] 96%|█████████▋| 385364/400000 [00:56<00:02, 6938.73it/s] 97%|█████████▋| 386089/400000 [00:56<00:01, 7028.56it/s] 97%|█████████▋| 386793/400000 [00:57<00:01, 6970.13it/s] 97%|█████████▋| 387491/400000 [00:57<00:01, 6962.99it/s] 97%|█████████▋| 388188/400000 [00:57<00:01, 6939.99it/s] 97%|█████████▋| 388905/400000 [00:57<00:01, 7006.30it/s] 97%|█████████▋| 389607/400000 [00:57<00:01, 6924.77it/s] 98%|█████████▊| 390325/400000 [00:57<00:01, 6996.63it/s] 98%|█████████▊| 391041/400000 [00:57<00:01, 7040.84it/s] 98%|█████████▊| 391746/400000 [00:57<00:01, 6976.81it/s] 98%|█████████▊| 392445/400000 [00:57<00:01, 6940.91it/s] 98%|█████████▊| 393171/400000 [00:57<00:00, 7033.50it/s] 98%|█████████▊| 393875/400000 [00:58<00:00, 6903.14it/s] 99%|█████████▊| 394595/400000 [00:58<00:00, 6987.90it/s] 99%|█████████▉| 395306/400000 [00:58<00:00, 7022.10it/s] 99%|█████████▉| 396009/400000 [00:58<00:00, 6965.70it/s] 99%|█████████▉| 396729/400000 [00:58<00:00, 7033.78it/s] 99%|█████████▉| 397433/400000 [00:58<00:00, 6969.71it/s]100%|█████████▉| 398131/400000 [00:58<00:00, 6749.98it/s]100%|█████████▉| 398808/400000 [00:58<00:00, 6704.00it/s]100%|█████████▉| 399530/400000 [00:58<00:00, 6849.96it/s]100%|█████████▉| 399999/400000 [00:58<00:00, 6780.81it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f32512bf6d8>, <torchtext.data.dataset.TabularDataset object at 0x7f32512bf828>, <torchtext.vocab.Vocab object at 0x7f32512bf748>), {}) 






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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00, 25.67 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.21 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00, 10.21 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.21 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  7.01 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.78 file/s]2020-05-30 16:37:19.288907: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-30 16:37:19.293881: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2397225000 Hz
2020-05-30 16:37:19.294248: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5646fcfd9e70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-30 16:37:19.294416: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'test', 'mnist2', 'cifar10', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:21, 121781.29it/s] 83%|████████▎ | 8241152/9912422 [00:00<00:09, 173861.49it/s]9920512it [00:00, 39980017.66it/s]                           
0it [00:00, ?it/s]32768it [00:00, 535882.88it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 151334.82it/s]1654784it [00:00, 11304165.45it/s]                         
0it [00:00, ?it/s]8192it [00:00, 173729.70it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

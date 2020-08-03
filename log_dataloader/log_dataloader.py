
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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5d3ed72b70> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5d3ed72b70> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f5da99e2510> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f5da99e2510> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5dc8c0fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5dc8c0fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5d56d8aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5d56d8aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5d56d8aa60> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:02, 158380.32it/s] 64%|██████▎   | 6299648/9912422 [00:00<00:15, 226012.89it/s]9920512it [00:00, 42165981.12it/s]                           
0it [00:00, ?it/s]32768it [00:00, 558853.96it/s]
0it [00:00, ?it/s]  3%|▎         | 49152/1648877 [00:00<00:03, 491309.62it/s]1654784it [00:00, 12378288.27it/s]                         
0it [00:00, ?it/s]8192it [00:00, 165872.08it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d3ec2e6a0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d3ec40908>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f5d56d8a6a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f5d56d8a6a8> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f5d56d8a6a8> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<02:11,  1.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<02:11,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<02:10,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<02:09,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<02:08,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<02:07,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<02:06,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<02:06,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<02:05,  1.23 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   6%|▌         | 9/162 [00:00<01:27,  1.74 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<01:27,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<01:27,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<01:26,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<01:26,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<01:25,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<01:24,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<01:24,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<01:23,  1.74 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:01<01:23,  1.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:01<01:22,  1.74 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  12%|█▏        | 19/162 [00:01<00:57,  2.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:01<00:57,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:01<00:57,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:01<00:57,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:01<00:56,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:01<00:56,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:01<00:55,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:01<00:55,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:01<00:55,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:01<00:54,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:01<00:54,  2.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  18%|█▊        | 29/162 [00:01<00:38,  3.49 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:01<00:38,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:01<00:37,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:01<00:37,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:01<00:37,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:01<00:36,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:01<00:36,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:36,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:36,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:35,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:35,  3.49 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  24%|██▍       | 39/162 [00:01<00:25,  4.91 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:25,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:24,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:24,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:24,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:24,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:24,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:23,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:23,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:23,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:23,  4.91 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.86 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:16,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:16,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:16,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:16,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:15,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:15,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:15,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:15,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:15,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:15,  6.86 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  36%|███▋      | 59/162 [00:01<00:10,  9.50 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:10,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:09,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:09,  9.50 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 13.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:07, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:07, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:06, 13.01 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.55 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:04, 17.55 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 23.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:03, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:03, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:03, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:03, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:02, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:02, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:02, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:02, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:02, 23.26 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 30.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:02, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:02, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:02, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 30.07 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 37.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:01, 37.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 46.14 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 46.14 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 54.42 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 54.42 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 62.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 69.25 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.60 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 75.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 75.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 75.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.60 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 75.60 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.54s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 75.60 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 32.75 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.95s/ url]
0 examples [00:00, ? examples/s]2020-08-03 12:11:00.917355: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-03 12:11:01.215686: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-03 12:11:01.215892: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55ac7cb4a290 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-03 12:11:01.215914: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  1.32 examples/s]107 examples [00:00,  1.89 examples/s]211 examples [00:00,  2.70 examples/s]313 examples [00:01,  3.85 examples/s]417 examples [00:01,  5.49 examples/s]523 examples [00:01,  7.82 examples/s]624 examples [00:01, 11.14 examples/s]718 examples [00:01, 15.83 examples/s]816 examples [00:01, 22.46 examples/s]918 examples [00:01, 31.79 examples/s]1022 examples [00:01, 44.82 examples/s]1120 examples [00:01, 62.65 examples/s]1222 examples [00:01, 87.18 examples/s]1323 examples [00:02, 120.06 examples/s]1428 examples [00:02, 163.46 examples/s]1528 examples [00:02, 215.34 examples/s]1628 examples [00:02, 281.64 examples/s]1731 examples [00:02, 359.94 examples/s]1830 examples [00:02, 444.78 examples/s]1929 examples [00:02, 527.36 examples/s]2032 examples [00:02, 617.00 examples/s]2136 examples [00:02, 702.18 examples/s]2236 examples [00:03, 760.19 examples/s]2338 examples [00:03, 822.41 examples/s]2443 examples [00:03, 877.74 examples/s]2548 examples [00:03, 921.16 examples/s]2654 examples [00:03, 958.20 examples/s]2757 examples [00:03, 963.44 examples/s]2859 examples [00:03, 974.08 examples/s]2960 examples [00:03, 984.25 examples/s]3061 examples [00:03, 989.00 examples/s]3162 examples [00:03, 990.67 examples/s]3263 examples [00:04, 988.14 examples/s]3364 examples [00:04, 994.51 examples/s]3469 examples [00:04, 1002.68 examples/s]3570 examples [00:04, 1002.43 examples/s]3671 examples [00:04, 991.70 examples/s] 3773 examples [00:04, 998.74 examples/s]3874 examples [00:04, 997.58 examples/s]3974 examples [00:04, 995.85 examples/s]4078 examples [00:04, 1008.13 examples/s]4181 examples [00:04, 1013.80 examples/s]4283 examples [00:05, 1002.08 examples/s]4384 examples [00:05, 971.66 examples/s] 4482 examples [00:05, 944.06 examples/s]4582 examples [00:05, 960.05 examples/s]4683 examples [00:05, 973.58 examples/s]4787 examples [00:05, 991.44 examples/s]4892 examples [00:05, 1005.82 examples/s]4994 examples [00:05, 1009.21 examples/s]5096 examples [00:05, 1010.13 examples/s]5199 examples [00:05, 1013.71 examples/s]5301 examples [00:06, 1012.86 examples/s]5405 examples [00:06, 1018.09 examples/s]5508 examples [00:06, 1021.47 examples/s]5611 examples [00:06, 943.03 examples/s] 5716 examples [00:06, 971.47 examples/s]5818 examples [00:06, 983.97 examples/s]5918 examples [00:06, 970.47 examples/s]6019 examples [00:06, 980.41 examples/s]6121 examples [00:06, 991.94 examples/s]6228 examples [00:07, 1013.51 examples/s]6337 examples [00:07, 1033.89 examples/s]6442 examples [00:07, 1037.21 examples/s]6546 examples [00:07, 1037.52 examples/s]6650 examples [00:07, 1006.99 examples/s]6756 examples [00:07, 1021.73 examples/s]6862 examples [00:07, 1030.12 examples/s]6966 examples [00:07, 1023.42 examples/s]7069 examples [00:07, 1013.87 examples/s]7171 examples [00:07, 982.01 examples/s] 7270 examples [00:08, 979.18 examples/s]7371 examples [00:08, 983.91 examples/s]7470 examples [00:08, 971.37 examples/s]7571 examples [00:08, 981.80 examples/s]7672 examples [00:08, 988.51 examples/s]7771 examples [00:08, 988.48 examples/s]7870 examples [00:08, 975.82 examples/s]7973 examples [00:08, 990.11 examples/s]8077 examples [00:08, 1003.17 examples/s]8178 examples [00:08, 999.76 examples/s] 8280 examples [00:09, 1003.39 examples/s]8383 examples [00:09, 1008.55 examples/s]8484 examples [00:09, 995.67 examples/s] 8588 examples [00:09, 1008.51 examples/s]8689 examples [00:09, 1003.78 examples/s]8791 examples [00:09, 1006.50 examples/s]8892 examples [00:09, 992.62 examples/s] 8992 examples [00:09, 993.82 examples/s]9092 examples [00:09, 991.42 examples/s]9194 examples [00:09, 999.43 examples/s]9294 examples [00:10, 992.98 examples/s]9394 examples [00:10, 989.49 examples/s]9493 examples [00:10, 971.32 examples/s]9592 examples [00:10, 976.34 examples/s]9690 examples [00:10, 953.95 examples/s]9790 examples [00:10, 965.89 examples/s]9897 examples [00:10, 994.26 examples/s]9998 examples [00:10, 998.47 examples/s]10099 examples [00:10, 952.96 examples/s]10202 examples [00:11, 974.73 examples/s]10301 examples [00:11, 961.00 examples/s]10405 examples [00:11, 981.47 examples/s]10504 examples [00:11, 979.68 examples/s]10608 examples [00:11, 995.46 examples/s]10712 examples [00:11, 1007.64 examples/s]10813 examples [00:11, 991.29 examples/s] 10916 examples [00:11, 1000.84 examples/s]11021 examples [00:11, 1012.75 examples/s]11123 examples [00:11, 1002.36 examples/s]11227 examples [00:12, 1011.04 examples/s]11330 examples [00:12, 1014.64 examples/s]11432 examples [00:12, 1013.22 examples/s]11534 examples [00:12, 1014.81 examples/s]11636 examples [00:12, 944.45 examples/s] 11740 examples [00:12, 969.78 examples/s]11842 examples [00:12, 980.86 examples/s]11941 examples [00:12, 924.89 examples/s]12044 examples [00:12, 951.91 examples/s]12141 examples [00:12, 942.42 examples/s]12249 examples [00:13, 977.79 examples/s]12354 examples [00:13, 992.23 examples/s]12454 examples [00:13, 991.13 examples/s]12555 examples [00:13, 990.29 examples/s]12655 examples [00:13, 979.72 examples/s]12754 examples [00:13, 979.35 examples/s]12856 examples [00:13, 988.76 examples/s]12957 examples [00:13, 994.20 examples/s]13057 examples [00:13, 990.84 examples/s]13157 examples [00:14, 977.66 examples/s]13262 examples [00:14, 996.79 examples/s]13367 examples [00:14, 1011.21 examples/s]13469 examples [00:14, 1011.25 examples/s]13571 examples [00:14, 976.79 examples/s] 13670 examples [00:14, 978.93 examples/s]13771 examples [00:14, 985.33 examples/s]13870 examples [00:14, 940.02 examples/s]13965 examples [00:14, 942.97 examples/s]14070 examples [00:14, 972.44 examples/s]14168 examples [00:15, 963.92 examples/s]14275 examples [00:15, 991.85 examples/s]14381 examples [00:15, 1011.02 examples/s]14488 examples [00:15, 1026.25 examples/s]14591 examples [00:15, 1023.29 examples/s]14694 examples [00:15, 1016.93 examples/s]14801 examples [00:15, 1028.67 examples/s]14905 examples [00:15, 1018.60 examples/s]15007 examples [00:15, 1009.25 examples/s]15109 examples [00:15, 988.35 examples/s] 15212 examples [00:16, 999.16 examples/s]15313 examples [00:16, 998.31 examples/s]15419 examples [00:16, 1014.47 examples/s]15521 examples [00:16, 987.60 examples/s] 15621 examples [00:16, 974.48 examples/s]15723 examples [00:16, 986.46 examples/s]15822 examples [00:16, 944.76 examples/s]15922 examples [00:16, 958.41 examples/s]16027 examples [00:16, 982.95 examples/s]16126 examples [00:17, 971.94 examples/s]16229 examples [00:17, 987.05 examples/s]16335 examples [00:17, 1006.34 examples/s]16440 examples [00:17, 1017.88 examples/s]16543 examples [00:17, 1019.93 examples/s]16647 examples [00:17, 1024.20 examples/s]16753 examples [00:17, 1033.75 examples/s]16861 examples [00:17, 1044.59 examples/s]16966 examples [00:17, 1034.05 examples/s]17070 examples [00:17, 1027.38 examples/s]17177 examples [00:18, 1037.78 examples/s]17281 examples [00:18, 1029.14 examples/s]17387 examples [00:18, 1037.30 examples/s]17491 examples [00:18, 1037.70 examples/s]17599 examples [00:18, 1048.22 examples/s]17704 examples [00:18, 1044.18 examples/s]17810 examples [00:18, 1046.42 examples/s]17915 examples [00:18, 1025.55 examples/s]18018 examples [00:18, 1009.90 examples/s]18120 examples [00:18, 1009.42 examples/s]18222 examples [00:19, 1012.26 examples/s]18328 examples [00:19, 1023.56 examples/s]18431 examples [00:19, 989.00 examples/s] 18531 examples [00:19, 972.41 examples/s]18630 examples [00:19, 975.78 examples/s]18731 examples [00:19, 983.30 examples/s]18833 examples [00:19, 992.42 examples/s]18935 examples [00:19, 997.87 examples/s]19040 examples [00:19, 1011.36 examples/s]19142 examples [00:19, 1013.58 examples/s]19244 examples [00:20, 1003.27 examples/s]19345 examples [00:20, 998.85 examples/s] 19452 examples [00:20, 1016.70 examples/s]19557 examples [00:20, 1026.44 examples/s]19660 examples [00:20, 1010.53 examples/s]19762 examples [00:20, 1007.33 examples/s]19866 examples [00:20, 1014.95 examples/s]19968 examples [00:20, 1005.79 examples/s]20069 examples [00:20, 909.58 examples/s] 20176 examples [00:21, 936.58 examples/s]20272 examples [00:21, 935.78 examples/s]20377 examples [00:21, 967.12 examples/s]20483 examples [00:21, 991.09 examples/s]20587 examples [00:21, 1004.53 examples/s]20690 examples [00:21, 1011.35 examples/s]20792 examples [00:21, 991.05 examples/s] 20893 examples [00:21, 995.86 examples/s]20998 examples [00:21, 1010.06 examples/s]21104 examples [00:21, 1022.53 examples/s]21210 examples [00:22, 1031.97 examples/s]21314 examples [00:22, 1025.79 examples/s]21417 examples [00:22, 1019.80 examples/s]21520 examples [00:22, 1009.20 examples/s]21624 examples [00:22, 1016.15 examples/s]21728 examples [00:22, 1020.28 examples/s]21831 examples [00:22, 1018.52 examples/s]21933 examples [00:22, 986.35 examples/s] 22032 examples [00:22, 947.39 examples/s]22128 examples [00:22, 948.93 examples/s]22231 examples [00:23, 969.27 examples/s]22329 examples [00:23, 951.99 examples/s]22425 examples [00:23, 953.88 examples/s]22528 examples [00:23, 973.95 examples/s]22631 examples [00:23, 989.81 examples/s]22731 examples [00:23, 985.89 examples/s]22830 examples [00:23, 965.44 examples/s]22927 examples [00:23, 949.63 examples/s]23024 examples [00:23, 953.14 examples/s]23120 examples [00:24, 940.94 examples/s]23220 examples [00:24, 956.59 examples/s]23320 examples [00:24, 967.08 examples/s]23422 examples [00:24, 980.64 examples/s]23521 examples [00:24, 966.58 examples/s]23618 examples [00:24, 956.43 examples/s]23717 examples [00:24, 964.46 examples/s]23814 examples [00:24, 950.41 examples/s]23910 examples [00:24, 943.66 examples/s]24008 examples [00:24, 954.02 examples/s]24105 examples [00:25, 958.65 examples/s]24201 examples [00:25, 951.13 examples/s]24299 examples [00:25, 957.30 examples/s]24399 examples [00:25, 968.73 examples/s]24501 examples [00:25, 983.07 examples/s]24604 examples [00:25, 995.96 examples/s]24704 examples [00:25, 980.75 examples/s]24803 examples [00:25, 981.84 examples/s]24909 examples [00:25, 1002.40 examples/s]25010 examples [00:25, 997.26 examples/s] 25110 examples [00:26, 987.05 examples/s]25212 examples [00:26, 994.45 examples/s]25312 examples [00:26, 968.31 examples/s]25415 examples [00:26, 985.81 examples/s]25515 examples [00:26, 988.74 examples/s]25615 examples [00:26, 986.27 examples/s]25717 examples [00:26, 993.55 examples/s]25823 examples [00:26, 1010.48 examples/s]25928 examples [00:26, 1021.40 examples/s]26031 examples [00:26, 1018.57 examples/s]26138 examples [00:27, 1031.56 examples/s]26242 examples [00:27, 1020.21 examples/s]26345 examples [00:27, 1010.00 examples/s]26449 examples [00:27, 1018.44 examples/s]26554 examples [00:27, 1027.21 examples/s]26661 examples [00:27, 1038.43 examples/s]26767 examples [00:27, 1043.80 examples/s]26872 examples [00:27, 1041.96 examples/s]26977 examples [00:27, 1040.68 examples/s]27082 examples [00:27, 1019.69 examples/s]27186 examples [00:28, 1025.41 examples/s]27291 examples [00:28, 1032.18 examples/s]27397 examples [00:28, 1037.61 examples/s]27503 examples [00:28, 1042.91 examples/s]27608 examples [00:28, 1031.43 examples/s]27712 examples [00:28, 1030.08 examples/s]27816 examples [00:28, 1025.30 examples/s]27919 examples [00:28, 982.13 examples/s] 28020 examples [00:28, 986.92 examples/s]28122 examples [00:29, 994.54 examples/s]28222 examples [00:29, 967.44 examples/s]28320 examples [00:29, 953.12 examples/s]28424 examples [00:29, 976.74 examples/s]28530 examples [00:29, 999.25 examples/s]28631 examples [00:29, 971.85 examples/s]28734 examples [00:29, 988.53 examples/s]28837 examples [00:29, 999.40 examples/s]28942 examples [00:29, 1011.67 examples/s]29044 examples [00:29, 1008.15 examples/s]29146 examples [00:30, 1010.10 examples/s]29248 examples [00:30, 1006.80 examples/s]29349 examples [00:30, 988.08 examples/s] 29455 examples [00:30, 1006.64 examples/s]29556 examples [00:30, 973.10 examples/s] 29654 examples [00:30, 931.52 examples/s]29756 examples [00:30, 956.11 examples/s]29860 examples [00:30, 978.06 examples/s]29959 examples [00:30, 957.92 examples/s]30056 examples [00:31, 915.57 examples/s]30162 examples [00:31, 954.25 examples/s]30269 examples [00:31, 986.01 examples/s]30375 examples [00:31, 1006.62 examples/s]30477 examples [00:31, 1005.75 examples/s]30583 examples [00:31, 1020.84 examples/s]30689 examples [00:31, 1029.53 examples/s]30793 examples [00:31, 995.69 examples/s] 30894 examples [00:31, 984.69 examples/s]30998 examples [00:31, 1000.30 examples/s]31099 examples [00:32, 989.97 examples/s] 31199 examples [00:32, 949.45 examples/s]31296 examples [00:32, 953.86 examples/s]31401 examples [00:32, 979.64 examples/s]31506 examples [00:32, 997.13 examples/s]31607 examples [00:32, 994.50 examples/s]31710 examples [00:32, 1003.68 examples/s]31817 examples [00:32, 1022.23 examples/s]31920 examples [00:32, 1000.42 examples/s]32021 examples [00:32, 993.11 examples/s] 32127 examples [00:33, 1011.09 examples/s]32232 examples [00:33, 1021.69 examples/s]32336 examples [00:33, 1024.81 examples/s]32439 examples [00:33, 1007.38 examples/s]32540 examples [00:33, 989.27 examples/s] 32640 examples [00:33, 991.77 examples/s]32740 examples [00:33, 973.32 examples/s]32841 examples [00:33, 983.32 examples/s]32942 examples [00:33, 989.21 examples/s]33047 examples [00:33, 1006.38 examples/s]33153 examples [00:34, 1020.61 examples/s]33258 examples [00:34, 1027.79 examples/s]33361 examples [00:34, 859.28 examples/s] 33452 examples [00:34, 801.53 examples/s]33537 examples [00:34, 806.33 examples/s]33639 examples [00:34, 860.36 examples/s]33741 examples [00:34, 902.00 examples/s]33835 examples [00:34, 912.45 examples/s]33929 examples [00:34, 902.39 examples/s]34021 examples [00:35, 867.42 examples/s]34109 examples [00:35, 865.88 examples/s]34207 examples [00:35, 895.33 examples/s]34298 examples [00:35, 808.15 examples/s]34382 examples [00:35, 764.40 examples/s]34484 examples [00:35, 824.73 examples/s]34586 examples [00:35, 873.55 examples/s]34688 examples [00:35, 910.66 examples/s]34795 examples [00:35, 951.16 examples/s]34900 examples [00:36, 978.72 examples/s]35002 examples [00:36, 989.30 examples/s]35106 examples [00:36, 1003.38 examples/s]35209 examples [00:36, 1010.40 examples/s]35314 examples [00:36, 1020.54 examples/s]35418 examples [00:36, 1024.03 examples/s]35521 examples [00:36, 1015.00 examples/s]35623 examples [00:36, 1005.95 examples/s]35724 examples [00:36, 992.94 examples/s] 35828 examples [00:36, 1005.23 examples/s]35929 examples [00:37, 975.94 examples/s] 36030 examples [00:37, 984.61 examples/s]36129 examples [00:37, 985.06 examples/s]36230 examples [00:37, 990.39 examples/s]36335 examples [00:37, 1005.09 examples/s]36438 examples [00:37, 1010.66 examples/s]36540 examples [00:37, 1003.68 examples/s]36647 examples [00:37, 1020.49 examples/s]36750 examples [00:37, 1009.38 examples/s]36857 examples [00:38, 1023.02 examples/s]36960 examples [00:38, 997.11 examples/s] 37060 examples [00:38, 952.81 examples/s]37165 examples [00:38, 978.13 examples/s]37267 examples [00:38, 987.70 examples/s]37370 examples [00:38, 998.44 examples/s]37471 examples [00:38, 996.60 examples/s]37576 examples [00:38, 1009.79 examples/s]37681 examples [00:38, 1019.80 examples/s]37787 examples [00:38, 1030.17 examples/s]37891 examples [00:39, 1029.55 examples/s]37997 examples [00:39, 1038.32 examples/s]38103 examples [00:39, 1042.20 examples/s]38208 examples [00:39, 1043.31 examples/s]38313 examples [00:39, 1042.20 examples/s]38418 examples [00:39, 1022.37 examples/s]38521 examples [00:39, 1023.45 examples/s]38624 examples [00:39, 1024.88 examples/s]38730 examples [00:39, 1033.62 examples/s]38839 examples [00:39, 1047.25 examples/s]38944 examples [00:40, 1038.69 examples/s]39048 examples [00:40, 1030.33 examples/s]39153 examples [00:40, 1034.74 examples/s]39257 examples [00:40, 1025.11 examples/s]39360 examples [00:40, 1010.05 examples/s]39464 examples [00:40, 1018.71 examples/s]39567 examples [00:40, 1021.42 examples/s]39671 examples [00:40, 1025.90 examples/s]39774 examples [00:40, 1025.99 examples/s]39877 examples [00:40, 1006.91 examples/s]39979 examples [00:41, 1010.74 examples/s]40081 examples [00:41, 954.30 examples/s] 40181 examples [00:41, 965.03 examples/s]40283 examples [00:41, 979.44 examples/s]40385 examples [00:41, 990.58 examples/s]40490 examples [00:41, 1005.19 examples/s]40591 examples [00:41, 997.28 examples/s] 40693 examples [00:41, 1003.62 examples/s]40796 examples [00:41, 1010.67 examples/s]40903 examples [00:42, 1024.91 examples/s]41007 examples [00:42, 1027.34 examples/s]41112 examples [00:42, 1032.38 examples/s]41218 examples [00:42, 1038.29 examples/s]41322 examples [00:42, 1033.59 examples/s]41427 examples [00:42, 1037.19 examples/s]41531 examples [00:42, 1010.02 examples/s]41633 examples [00:42, 1000.86 examples/s]41740 examples [00:42, 1019.15 examples/s]41843 examples [00:42, 1007.99 examples/s]41944 examples [00:43, 972.06 examples/s] 42044 examples [00:43, 977.32 examples/s]42145 examples [00:43, 984.51 examples/s]42244 examples [00:43, 981.00 examples/s]42348 examples [00:43, 995.72 examples/s]42449 examples [00:43, 997.31 examples/s]42554 examples [00:43, 1011.83 examples/s]42657 examples [00:43, 1015.02 examples/s]42760 examples [00:43, 1019.09 examples/s]42866 examples [00:43, 1028.11 examples/s]42974 examples [00:44, 1039.88 examples/s]43079 examples [00:44, 1022.93 examples/s]43184 examples [00:44, 1030.79 examples/s]43292 examples [00:44, 1043.93 examples/s]43397 examples [00:44, 1044.81 examples/s]43502 examples [00:44, 1039.25 examples/s]43608 examples [00:44, 1043.36 examples/s]43713 examples [00:44, 1043.18 examples/s]43818 examples [00:44, 1041.75 examples/s]43923 examples [00:44, 1036.27 examples/s]44027 examples [00:45, 1035.08 examples/s]44132 examples [00:45, 1038.02 examples/s]44236 examples [00:45, 1019.61 examples/s]44341 examples [00:45, 1027.12 examples/s]44444 examples [00:45, 1012.73 examples/s]44550 examples [00:45, 1023.66 examples/s]44656 examples [00:45, 1032.11 examples/s]44760 examples [00:45, 1012.72 examples/s]44866 examples [00:45, 1025.24 examples/s]44970 examples [00:45, 1027.81 examples/s]45075 examples [00:46, 1032.33 examples/s]45181 examples [00:46, 1038.98 examples/s]45285 examples [00:46, 1037.79 examples/s]45389 examples [00:46, 1016.36 examples/s]45491 examples [00:46, 955.28 examples/s] 45595 examples [00:46, 976.98 examples/s]45694 examples [00:46, 947.32 examples/s]45790 examples [00:46, 949.95 examples/s]45888 examples [00:46, 956.76 examples/s]45989 examples [00:47, 971.45 examples/s]46088 examples [00:47, 974.93 examples/s]46190 examples [00:47, 987.56 examples/s]46296 examples [00:47, 1008.22 examples/s]46398 examples [00:47, 1005.80 examples/s]46499 examples [00:47, 1006.77 examples/s]46601 examples [00:47, 1009.71 examples/s]46706 examples [00:47, 1020.58 examples/s]46812 examples [00:47, 1031.70 examples/s]46916 examples [00:47, 1025.80 examples/s]47020 examples [00:48, 1029.94 examples/s]47126 examples [00:48, 1038.03 examples/s]47236 examples [00:48, 1053.87 examples/s]47343 examples [00:48, 1056.81 examples/s]47449 examples [00:48, 1031.63 examples/s]47553 examples [00:48, 1015.74 examples/s]47659 examples [00:48, 1027.80 examples/s]47762 examples [00:48, 989.56 examples/s] 47867 examples [00:48, 1006.89 examples/s]47971 examples [00:48, 1014.02 examples/s]48076 examples [00:49, 1020.65 examples/s]48183 examples [00:49, 1031.91 examples/s]48287 examples [00:49, 1023.55 examples/s]48391 examples [00:49, 1027.03 examples/s]48495 examples [00:49, 1030.45 examples/s]48599 examples [00:49, 1029.60 examples/s]48705 examples [00:49, 1037.93 examples/s]48809 examples [00:49, 1024.36 examples/s]48915 examples [00:49, 1033.95 examples/s]49021 examples [00:49, 1039.24 examples/s]49125 examples [00:50, 1010.85 examples/s]49227 examples [00:50, 980.47 examples/s] 49332 examples [00:50, 999.04 examples/s]49435 examples [00:50, 1007.46 examples/s]49538 examples [00:50, 1013.14 examples/s]49640 examples [00:50, 1012.07 examples/s]49742 examples [00:50, 1008.96 examples/s]49849 examples [00:50, 1026.34 examples/s]49953 examples [00:50, 1030.23 examples/s]                                            0%|          | 0/50000 [00:00<?, ? examples/s] 13%|█▎        | 6657/50000 [00:00<00:00, 66568.48 examples/s] 38%|███▊      | 19246/50000 [00:00<00:00, 77528.01 examples/s] 63%|██████▎   | 31491/50000 [00:00<00:00, 87115.73 examples/s] 89%|████████▉ | 44583/50000 [00:00<00:00, 96833.83 examples/s]                                                               0 examples [00:00, ? examples/s]80 examples [00:00, 796.52 examples/s]181 examples [00:00, 848.97 examples/s]275 examples [00:00, 872.76 examples/s]383 examples [00:00, 923.98 examples/s]489 examples [00:00, 959.51 examples/s]596 examples [00:00, 988.28 examples/s]703 examples [00:00, 1008.89 examples/s]807 examples [00:00, 1017.51 examples/s]915 examples [00:00, 1034.56 examples/s]1020 examples [00:01, 1036.89 examples/s]1126 examples [00:01, 1041.10 examples/s]1231 examples [00:01, 1041.37 examples/s]1335 examples [00:01, 1029.24 examples/s]1439 examples [00:01, 1032.33 examples/s]1542 examples [00:01, 1002.17 examples/s]1648 examples [00:01, 1016.74 examples/s]1752 examples [00:01, 1023.15 examples/s]1855 examples [00:01, 1017.31 examples/s]1960 examples [00:01, 1026.33 examples/s]2066 examples [00:02, 1032.89 examples/s]2173 examples [00:02, 1042.83 examples/s]2278 examples [00:02, 1044.08 examples/s]2383 examples [00:02, 1041.60 examples/s]2491 examples [00:02, 1052.29 examples/s]2597 examples [00:02, 1052.90 examples/s]2704 examples [00:02, 1057.89 examples/s]2810 examples [00:02, 1056.22 examples/s]2916 examples [00:02, 1046.42 examples/s]3022 examples [00:02, 1049.89 examples/s]3128 examples [00:03, 1050.09 examples/s]3234 examples [00:03, 1051.52 examples/s]3340 examples [00:03, 1052.44 examples/s]3446 examples [00:03, 1048.37 examples/s]3551 examples [00:03, 1047.88 examples/s]3657 examples [00:03, 1051.24 examples/s]3763 examples [00:03, 1045.38 examples/s]3868 examples [00:03, 1041.63 examples/s]3973 examples [00:03, 1042.36 examples/s]4081 examples [00:03, 1050.85 examples/s]4187 examples [00:04, 1040.47 examples/s]4293 examples [00:04, 1045.91 examples/s]4398 examples [00:04, 1046.17 examples/s]4504 examples [00:04, 1049.92 examples/s]4612 examples [00:04, 1057.78 examples/s]4721 examples [00:04, 1064.42 examples/s]4828 examples [00:04, 1064.74 examples/s]4935 examples [00:04, 1057.50 examples/s]5041 examples [00:04, 1028.06 examples/s]5149 examples [00:04, 1040.56 examples/s]5255 examples [00:05, 1045.45 examples/s]5360 examples [00:05, 1026.37 examples/s]5463 examples [00:05, 1013.74 examples/s]5568 examples [00:05, 1022.09 examples/s]5673 examples [00:05, 1027.44 examples/s]5778 examples [00:05, 1033.75 examples/s]5887 examples [00:05, 1049.11 examples/s]5993 examples [00:05, 1052.10 examples/s]6100 examples [00:05, 1054.85 examples/s]6207 examples [00:05, 1056.87 examples/s]6315 examples [00:06, 1063.24 examples/s]6423 examples [00:06, 1067.79 examples/s]6530 examples [00:06, 1063.72 examples/s]6638 examples [00:06, 1066.85 examples/s]6745 examples [00:06, 1067.63 examples/s]6855 examples [00:06, 1075.89 examples/s]6965 examples [00:06, 1080.71 examples/s]7074 examples [00:06, 1070.06 examples/s]7182 examples [00:06, 1061.02 examples/s]7290 examples [00:06, 1065.28 examples/s]7397 examples [00:07, 1050.43 examples/s]7503 examples [00:07, 1037.75 examples/s]7607 examples [00:07, 1038.39 examples/s]7711 examples [00:07, 1014.43 examples/s]7813 examples [00:07, 997.85 examples/s] 7919 examples [00:07, 1013.26 examples/s]8024 examples [00:07, 1022.88 examples/s]8127 examples [00:07, 1013.53 examples/s]8229 examples [00:07, 973.37 examples/s] 8327 examples [00:08, 968.87 examples/s]8425 examples [00:08, 969.93 examples/s]8529 examples [00:08, 988.78 examples/s]8633 examples [00:08, 1003.59 examples/s]8740 examples [00:08, 1020.64 examples/s]8846 examples [00:08, 1030.68 examples/s]8952 examples [00:08, 1037.81 examples/s]9056 examples [00:08, 1035.95 examples/s]9160 examples [00:08, 1014.39 examples/s]9262 examples [00:08, 1010.52 examples/s]9364 examples [00:09, 1011.83 examples/s]9470 examples [00:09, 1024.68 examples/s]9573 examples [00:09, 1015.10 examples/s]9679 examples [00:09, 1026.80 examples/s]9785 examples [00:09, 1035.82 examples/s]9889 examples [00:09, 1035.94 examples/s]9996 examples [00:09, 1045.14 examples/s]                                           0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNIGRDQ/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNIGRDQ/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5d56d8aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5d56d8aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5d56d8aa60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5da26f70b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5ce10104a8>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f5d56d8aa60> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f5d56d8aa60> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f5d56d8aa60> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d3ec402e8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5d3ec40198>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.49 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  6.49 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  6.49 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.83 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00,  7.83 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.52 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.50 file/s]2020-08-03 12:12:10.121142: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 AVX512F FMA
2020-08-03 12:12:10.125624: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2095074999 Hz
2020-08-03 12:12:10.125980: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x56110267db70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-08-03 12:12:10.126000: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['train', 'cifar10', 'mnist2', 'test', 'mnist_dataset_small.npy', 'fashion-mnist_small.npy'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:05, 151619.30it/s] 76%|███████▌  | 7553024/9912422 [00:00<00:10, 216412.23it/s]9920512it [00:00, 43254391.40it/s]                           
0it [00:00, ?it/s]32768it [00:00, 630242.87it/s]
0it [00:00, ?it/s]  6%|▋         | 106496/1648877 [00:00<00:01, 987947.16it/s]1654784it [00:00, 12506405.13it/s]                          
0it [00:00, ?it/s]8192it [00:00, 186210.45it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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


  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/4cb29075603a9201327b06e3ac95aac9279ebf78', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': '4cb29075603a9201327b06e3ac95aac9279ebf78', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/4cb29075603a9201327b06e3ac95aac9279ebf78

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/4cb29075603a9201327b06e3ac95aac9279ebf78

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/4cb29075603a9201327b06e3ac95aac9279ebf78

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f1e5dbb1ae8> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f1e5dbb1ae8> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f1ec8823488> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f1ec8823488> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f1ee7a4fea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f1ee7a4fea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1e75bca9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1e75bca9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1e75bca9d8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 26%|██▌       | 2547712/9912422 [00:00<00:00, 25471921.53it/s]9920512it [00:00, 30003009.10it/s]                             
0it [00:00, ?it/s]32768it [00:00, 587002.28it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 148707.08it/s]1654784it [00:00, 11542692.47it/s]                         
0it [00:00, ?it/s]8192it [00:00, 122076.37it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1e545ef7b8>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1e545fda20>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f1e75bca620> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f1e75bca620> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f1e75bca620> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:19,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:18,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<01:17,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<01:16,  2.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.85 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:54,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:53,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:52,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:51,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:51,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:51,  2.85 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  4.02 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:36,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:35,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:34,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:33,  4.02 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:23,  5.63 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
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

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:00<00:22,  5.63 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.84 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:00<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:15,  7.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:14,  7.84 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.78 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:10, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:09, 10.78 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:07, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:06, 14.65 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.51 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:04, 19.51 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:03, 25.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.47 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:02, 32.47 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.25 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:01<00:01, 40.25 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.48 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:01<00:01, 48.48 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:01<00:00, 56.57 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 63.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:01<00:00, 63.79 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.40 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 69.40 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.36 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 74.36 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.16 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:02<00:00, 78.16 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.83 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]Dl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:02<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:02<00:00, 81.83 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:02<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.55s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:04<00:00,  2.32s/ url]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 81.83 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.55s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ file]
Dl Size...: 100%|██████████| 162/162 [00:04<00:00, 35.55 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:04<00:00,  4.56s/ url]
0 examples [00:00, ? examples/s]2020-07-26 14:22:06.463571: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-26 14:22:06.468652: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-26 14:22:06.468798: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5629ea931310 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-26 14:22:06.468815: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
64 examples [00:00, 637.58 examples/s]164 examples [00:00, 714.86 examples/s]254 examples [00:00, 761.10 examples/s]353 examples [00:00, 816.69 examples/s]449 examples [00:00, 854.42 examples/s]547 examples [00:00, 886.89 examples/s]646 examples [00:00, 914.25 examples/s]737 examples [00:00, 912.73 examples/s]832 examples [00:00, 922.59 examples/s]927 examples [00:01, 930.25 examples/s]1024 examples [00:01, 940.61 examples/s]1118 examples [00:01, 935.68 examples/s]1219 examples [00:01, 956.05 examples/s]1318 examples [00:01, 964.67 examples/s]1415 examples [00:01, 947.93 examples/s]1510 examples [00:01, 931.18 examples/s]1604 examples [00:01, 922.67 examples/s]1697 examples [00:01, 899.65 examples/s]1797 examples [00:01, 925.17 examples/s]1891 examples [00:02, 928.58 examples/s]1985 examples [00:02, 931.57 examples/s]2084 examples [00:02, 947.37 examples/s]2181 examples [00:02, 953.50 examples/s]2277 examples [00:02, 940.39 examples/s]2372 examples [00:02, 930.30 examples/s]2466 examples [00:02, 926.88 examples/s]2559 examples [00:02, 915.57 examples/s]2659 examples [00:02, 936.74 examples/s]2754 examples [00:02, 939.65 examples/s]2849 examples [00:03, 935.57 examples/s]2948 examples [00:03, 950.57 examples/s]3044 examples [00:03, 951.31 examples/s]3141 examples [00:03, 954.79 examples/s]3241 examples [00:03, 967.19 examples/s]3338 examples [00:03, 961.95 examples/s]3438 examples [00:03, 972.52 examples/s]3539 examples [00:03, 983.31 examples/s]3638 examples [00:03, 983.41 examples/s]3737 examples [00:03, 966.69 examples/s]3834 examples [00:04, 944.28 examples/s]3929 examples [00:04, 944.24 examples/s]4024 examples [00:04, 926.36 examples/s]4122 examples [00:04, 939.24 examples/s]4226 examples [00:04, 965.62 examples/s]4323 examples [00:04, 951.23 examples/s]4423 examples [00:04, 963.73 examples/s]4522 examples [00:04, 970.74 examples/s]4623 examples [00:04, 980.52 examples/s]4722 examples [00:04, 973.00 examples/s]4820 examples [00:05, 952.77 examples/s]4918 examples [00:05, 958.11 examples/s]5018 examples [00:05, 969.11 examples/s]5120 examples [00:05, 982.09 examples/s]5221 examples [00:05, 987.77 examples/s]5320 examples [00:05, 926.72 examples/s]5417 examples [00:05, 938.47 examples/s]5514 examples [00:05, 945.00 examples/s]5609 examples [00:05, 939.70 examples/s]5704 examples [00:06, 935.94 examples/s]5798 examples [00:06, 934.30 examples/s]5893 examples [00:06, 938.32 examples/s]5995 examples [00:06, 960.03 examples/s]6100 examples [00:06, 985.20 examples/s]6202 examples [00:06, 994.10 examples/s]6302 examples [00:06, 982.42 examples/s]6401 examples [00:06, 971.50 examples/s]6500 examples [00:06, 976.25 examples/s]6598 examples [00:06, 970.22 examples/s]6698 examples [00:07, 977.74 examples/s]6796 examples [00:07, 953.58 examples/s]6892 examples [00:07, 945.40 examples/s]6988 examples [00:07, 949.04 examples/s]7084 examples [00:07, 939.23 examples/s]7184 examples [00:07, 953.98 examples/s]7281 examples [00:07, 958.59 examples/s]7381 examples [00:07, 969.81 examples/s]7479 examples [00:07, 955.93 examples/s]7575 examples [00:07, 950.27 examples/s]7671 examples [00:08, 936.28 examples/s]7765 examples [00:08, 930.36 examples/s]7862 examples [00:08, 940.90 examples/s]7960 examples [00:08, 950.51 examples/s]8056 examples [00:08, 929.59 examples/s]8155 examples [00:08, 945.54 examples/s]8250 examples [00:08, 933.74 examples/s]8344 examples [00:08, 928.07 examples/s]8442 examples [00:08, 941.10 examples/s]8541 examples [00:09, 954.55 examples/s]8637 examples [00:09, 948.98 examples/s]8733 examples [00:09, 947.39 examples/s]8828 examples [00:09, 933.74 examples/s]8923 examples [00:09, 934.49 examples/s]9017 examples [00:09, 934.38 examples/s]9116 examples [00:09, 950.14 examples/s]9212 examples [00:09, 940.74 examples/s]9308 examples [00:09, 945.13 examples/s]9403 examples [00:09, 938.53 examples/s]9497 examples [00:10, 936.59 examples/s]9593 examples [00:10, 942.16 examples/s]9688 examples [00:10, 928.97 examples/s]9783 examples [00:10, 932.88 examples/s]9879 examples [00:10, 938.10 examples/s]9973 examples [00:10, 926.65 examples/s]10066 examples [00:10, 876.70 examples/s]10162 examples [00:10, 899.70 examples/s]10260 examples [00:10, 921.50 examples/s]10358 examples [00:10, 937.02 examples/s]10455 examples [00:11, 946.19 examples/s]10554 examples [00:11, 958.73 examples/s]10651 examples [00:11, 948.53 examples/s]10751 examples [00:11, 960.68 examples/s]10851 examples [00:11, 970.52 examples/s]10950 examples [00:11, 975.63 examples/s]11048 examples [00:11, 963.97 examples/s]11146 examples [00:11, 966.67 examples/s]11247 examples [00:11, 978.00 examples/s]11347 examples [00:11, 982.09 examples/s]11446 examples [00:12, 963.42 examples/s]11543 examples [00:12, 947.97 examples/s]11640 examples [00:12, 954.41 examples/s]11742 examples [00:12, 972.08 examples/s]11840 examples [00:12, 959.79 examples/s]11937 examples [00:12, 961.02 examples/s]12036 examples [00:12, 967.22 examples/s]12133 examples [00:12, 957.86 examples/s]12230 examples [00:12, 960.42 examples/s]12327 examples [00:12, 955.84 examples/s]12426 examples [00:13, 965.47 examples/s]12527 examples [00:13, 976.08 examples/s]12625 examples [00:13, 969.83 examples/s]12726 examples [00:13, 978.89 examples/s]12825 examples [00:13, 979.83 examples/s]12924 examples [00:13, 975.61 examples/s]13022 examples [00:13, 968.37 examples/s]13119 examples [00:13, 947.43 examples/s]13217 examples [00:13, 956.44 examples/s]13313 examples [00:14, 947.63 examples/s]13409 examples [00:14, 948.52 examples/s]13504 examples [00:14, 945.35 examples/s]13599 examples [00:14, 941.08 examples/s]13697 examples [00:14, 952.34 examples/s]13795 examples [00:14, 957.58 examples/s]13891 examples [00:14, 897.91 examples/s]13982 examples [00:14, 898.07 examples/s]14073 examples [00:14, 898.89 examples/s]14167 examples [00:14, 908.44 examples/s]14264 examples [00:15, 923.13 examples/s]14363 examples [00:15, 940.84 examples/s]14458 examples [00:15, 927.49 examples/s]14551 examples [00:15, 924.38 examples/s]14649 examples [00:15, 940.06 examples/s]14746 examples [00:15, 947.09 examples/s]14841 examples [00:15, 945.35 examples/s]14936 examples [00:15, 937.65 examples/s]15035 examples [00:15, 951.19 examples/s]15131 examples [00:15, 930.18 examples/s]15226 examples [00:16, 934.22 examples/s]15320 examples [00:16, 934.06 examples/s]15418 examples [00:16, 945.59 examples/s]15513 examples [00:16, 944.95 examples/s]15613 examples [00:16, 960.13 examples/s]15713 examples [00:16, 971.74 examples/s]15811 examples [00:16, 959.99 examples/s]15908 examples [00:16, 937.73 examples/s]16004 examples [00:16, 943.69 examples/s]16105 examples [00:16, 959.88 examples/s]16202 examples [00:17, 957.37 examples/s]16303 examples [00:17, 970.65 examples/s]16403 examples [00:17, 978.01 examples/s]16504 examples [00:17, 985.62 examples/s]16605 examples [00:17, 992.79 examples/s]16705 examples [00:17, 987.42 examples/s]16804 examples [00:17, 980.47 examples/s]16903 examples [00:17, 974.48 examples/s]17001 examples [00:17, 955.43 examples/s]17097 examples [00:18, 935.63 examples/s]17192 examples [00:18, 937.92 examples/s]17286 examples [00:18, 920.16 examples/s]17379 examples [00:18, 867.25 examples/s]17474 examples [00:18, 888.88 examples/s]17573 examples [00:18, 916.86 examples/s]17666 examples [00:18, 900.29 examples/s]17766 examples [00:18, 927.03 examples/s]17860 examples [00:18, 930.68 examples/s]17961 examples [00:18, 952.83 examples/s]18061 examples [00:19, 964.49 examples/s]18160 examples [00:19, 969.76 examples/s]18260 examples [00:19, 978.32 examples/s]18359 examples [00:19, 964.58 examples/s]18458 examples [00:19, 971.12 examples/s]18556 examples [00:19, 973.55 examples/s]18654 examples [00:19, 958.30 examples/s]18750 examples [00:19, 906.25 examples/s]18845 examples [00:19, 918.45 examples/s]18938 examples [00:20, 865.70 examples/s]19032 examples [00:20, 884.43 examples/s]19135 examples [00:20, 921.76 examples/s]19237 examples [00:20, 947.97 examples/s]19338 examples [00:20, 963.13 examples/s]19439 examples [00:20, 974.94 examples/s]19537 examples [00:20, 976.18 examples/s]19635 examples [00:20, 969.34 examples/s]19733 examples [00:20, 968.42 examples/s]19833 examples [00:20, 974.83 examples/s]19933 examples [00:21, 980.72 examples/s]20032 examples [00:21, 918.87 examples/s]20130 examples [00:21, 934.25 examples/s]20226 examples [00:21, 939.52 examples/s]20322 examples [00:21, 944.52 examples/s]20417 examples [00:21, 925.17 examples/s]20510 examples [00:21, 884.01 examples/s]20600 examples [00:21, 863.30 examples/s]20696 examples [00:21, 889.49 examples/s]20793 examples [00:21, 909.96 examples/s]20890 examples [00:22, 925.73 examples/s]20988 examples [00:22, 940.35 examples/s]21086 examples [00:22, 947.91 examples/s]21182 examples [00:22, 945.23 examples/s]21277 examples [00:22, 942.62 examples/s]21374 examples [00:22, 950.28 examples/s]21470 examples [00:22, 949.81 examples/s]21567 examples [00:22, 952.90 examples/s]21667 examples [00:22, 963.93 examples/s]21766 examples [00:22, 971.19 examples/s]21864 examples [00:23, 937.69 examples/s]21959 examples [00:23, 922.27 examples/s]22052 examples [00:23, 919.47 examples/s]22145 examples [00:23, 917.90 examples/s]22237 examples [00:23, 916.45 examples/s]22329 examples [00:23, 911.31 examples/s]22421 examples [00:23, 886.88 examples/s]22518 examples [00:23, 909.83 examples/s]22614 examples [00:23, 921.91 examples/s]22712 examples [00:24, 937.88 examples/s]22808 examples [00:24, 942.86 examples/s]22908 examples [00:24, 956.94 examples/s]23004 examples [00:24, 948.80 examples/s]23100 examples [00:24, 926.06 examples/s]23198 examples [00:24, 938.91 examples/s]23293 examples [00:24, 936.89 examples/s]23390 examples [00:24, 945.55 examples/s]23485 examples [00:24, 899.12 examples/s]23576 examples [00:24, 855.29 examples/s]23666 examples [00:25, 867.76 examples/s]23761 examples [00:25, 890.78 examples/s]23859 examples [00:25, 914.83 examples/s]23954 examples [00:25, 922.60 examples/s]24047 examples [00:25, 910.82 examples/s]24139 examples [00:25, 906.37 examples/s]24231 examples [00:25, 909.56 examples/s]24323 examples [00:25, 904.69 examples/s]24417 examples [00:25, 912.46 examples/s]24509 examples [00:25, 907.79 examples/s]24604 examples [00:26, 919.91 examples/s]24701 examples [00:26, 933.96 examples/s]24801 examples [00:26, 952.25 examples/s]24897 examples [00:26, 946.04 examples/s]24992 examples [00:26, 943.96 examples/s]25087 examples [00:26, 929.96 examples/s]25185 examples [00:26, 941.46 examples/s]25280 examples [00:26, 941.40 examples/s]25380 examples [00:26, 956.16 examples/s]25476 examples [00:27, 949.60 examples/s]25575 examples [00:27, 958.78 examples/s]25675 examples [00:27, 970.36 examples/s]25774 examples [00:27, 975.30 examples/s]25874 examples [00:27, 979.80 examples/s]25973 examples [00:27, 964.61 examples/s]26070 examples [00:27, 964.79 examples/s]26167 examples [00:27, 957.83 examples/s]26263 examples [00:27, 947.50 examples/s]26358 examples [00:27, 933.27 examples/s]26452 examples [00:28, 925.67 examples/s]26550 examples [00:28, 939.02 examples/s]26651 examples [00:28, 958.12 examples/s]26751 examples [00:28, 969.15 examples/s]26850 examples [00:28, 975.21 examples/s]26948 examples [00:28, 968.27 examples/s]27045 examples [00:28, 968.37 examples/s]27142 examples [00:28, 958.94 examples/s]27243 examples [00:28, 972.58 examples/s]27341 examples [00:28, 974.70 examples/s]27439 examples [00:29, 968.03 examples/s]27536 examples [00:29, 961.40 examples/s]27633 examples [00:29, 950.25 examples/s]27734 examples [00:29, 966.73 examples/s]27836 examples [00:29, 979.01 examples/s]27935 examples [00:29, 975.53 examples/s]28033 examples [00:29, 955.57 examples/s]28134 examples [00:29, 969.67 examples/s]28236 examples [00:29, 981.52 examples/s]28335 examples [00:29, 976.59 examples/s]28433 examples [00:30, 956.61 examples/s]28529 examples [00:30, 916.08 examples/s]28622 examples [00:30, 917.53 examples/s]28719 examples [00:30, 930.71 examples/s]28813 examples [00:30, 926.30 examples/s]28906 examples [00:30, 916.69 examples/s]29001 examples [00:30, 925.79 examples/s]29094 examples [00:30, 879.21 examples/s]29197 examples [00:30, 917.24 examples/s]29292 examples [00:31, 925.04 examples/s]29386 examples [00:31, 892.75 examples/s]29487 examples [00:31, 923.54 examples/s]29581 examples [00:31, 919.22 examples/s]29677 examples [00:31, 929.20 examples/s]29771 examples [00:31, 912.67 examples/s]29864 examples [00:31, 916.08 examples/s]29960 examples [00:31, 926.34 examples/s]30053 examples [00:31, 867.81 examples/s]30152 examples [00:31, 900.03 examples/s]30243 examples [00:32, 902.54 examples/s]30342 examples [00:32, 925.74 examples/s]30436 examples [00:32, 921.32 examples/s]30529 examples [00:32, 922.75 examples/s]30627 examples [00:32, 937.52 examples/s]30722 examples [00:32, 934.32 examples/s]30817 examples [00:32, 937.21 examples/s]30915 examples [00:32, 947.61 examples/s]31010 examples [00:32, 939.43 examples/s]31105 examples [00:32, 940.11 examples/s]31200 examples [00:33, 935.11 examples/s]31300 examples [00:33, 952.06 examples/s]31397 examples [00:33, 955.42 examples/s]31497 examples [00:33, 967.22 examples/s]31594 examples [00:33, 956.79 examples/s]31690 examples [00:33, 956.28 examples/s]31786 examples [00:33, 956.54 examples/s]31882 examples [00:33, 950.71 examples/s]31978 examples [00:33, 944.37 examples/s]32074 examples [00:34, 948.39 examples/s]32169 examples [00:34, 943.09 examples/s]32267 examples [00:34, 953.72 examples/s]32363 examples [00:34, 953.14 examples/s]32459 examples [00:34, 953.81 examples/s]32555 examples [00:34, 946.24 examples/s]32650 examples [00:34, 937.83 examples/s]32745 examples [00:34, 939.17 examples/s]32840 examples [00:34, 942.33 examples/s]32936 examples [00:34, 945.12 examples/s]33032 examples [00:35, 946.89 examples/s]33127 examples [00:35, 945.43 examples/s]33223 examples [00:35, 947.33 examples/s]33318 examples [00:35, 931.28 examples/s]33412 examples [00:35, 922.84 examples/s]33511 examples [00:35, 939.71 examples/s]33606 examples [00:35, 932.03 examples/s]33700 examples [00:35, 915.53 examples/s]33796 examples [00:35, 926.41 examples/s]33896 examples [00:35, 944.59 examples/s]33991 examples [00:36, 936.88 examples/s]34085 examples [00:36, 913.78 examples/s]34184 examples [00:36, 934.19 examples/s]34283 examples [00:36, 950.23 examples/s]34384 examples [00:36, 966.95 examples/s]34484 examples [00:36, 973.37 examples/s]34582 examples [00:36, 966.19 examples/s]34682 examples [00:36, 973.56 examples/s]34780 examples [00:36, 947.69 examples/s]34875 examples [00:36, 945.42 examples/s]34970 examples [00:37, 943.72 examples/s]35065 examples [00:37, 929.84 examples/s]35165 examples [00:37, 948.46 examples/s]35262 examples [00:37, 953.79 examples/s]35358 examples [00:37, 938.07 examples/s]35452 examples [00:37, 925.41 examples/s]35545 examples [00:37, 923.55 examples/s]35642 examples [00:37, 934.94 examples/s]35741 examples [00:37, 948.39 examples/s]35836 examples [00:37, 947.17 examples/s]35932 examples [00:38, 950.26 examples/s]36028 examples [00:38, 940.80 examples/s]36128 examples [00:38, 956.74 examples/s]36226 examples [00:38, 963.42 examples/s]36323 examples [00:38, 956.85 examples/s]36421 examples [00:38, 961.22 examples/s]36518 examples [00:38, 954.64 examples/s]36617 examples [00:38, 963.21 examples/s]36718 examples [00:38, 975.23 examples/s]36816 examples [00:39, 976.01 examples/s]36914 examples [00:39, 968.02 examples/s]37011 examples [00:39, 949.95 examples/s]37107 examples [00:39, 917.40 examples/s]37204 examples [00:39, 931.57 examples/s]37303 examples [00:39, 945.97 examples/s]37398 examples [00:39, 939.09 examples/s]37494 examples [00:39, 944.10 examples/s]37591 examples [00:39, 948.47 examples/s]37688 examples [00:39, 953.68 examples/s]37784 examples [00:40, 944.43 examples/s]37879 examples [00:40, 938.28 examples/s]37973 examples [00:40, 931.91 examples/s]38068 examples [00:40, 937.00 examples/s]38165 examples [00:40, 943.98 examples/s]38260 examples [00:40, 932.31 examples/s]38354 examples [00:40, 924.30 examples/s]38452 examples [00:40, 938.03 examples/s]38546 examples [00:40, 936.49 examples/s]38642 examples [00:40, 940.29 examples/s]38742 examples [00:41, 956.19 examples/s]38838 examples [00:41, 956.07 examples/s]38936 examples [00:41, 961.10 examples/s]39034 examples [00:41, 966.06 examples/s]39133 examples [00:41, 971.38 examples/s]39231 examples [00:41, 954.21 examples/s]39327 examples [00:41, 943.90 examples/s]39424 examples [00:41, 950.42 examples/s]39522 examples [00:41, 956.37 examples/s]39620 examples [00:41, 960.96 examples/s]39717 examples [00:42, 959.20 examples/s]39813 examples [00:42, 954.74 examples/s]39916 examples [00:42, 974.34 examples/s]40014 examples [00:42, 903.18 examples/s]40106 examples [00:42, 896.12 examples/s]40197 examples [00:42, 897.78 examples/s]40291 examples [00:42, 908.98 examples/s]40388 examples [00:42, 924.21 examples/s]40486 examples [00:42, 940.21 examples/s]40581 examples [00:43, 937.45 examples/s]40681 examples [00:43, 954.93 examples/s]40777 examples [00:43, 946.13 examples/s]40875 examples [00:43, 954.96 examples/s]40973 examples [00:43, 961.38 examples/s]41070 examples [00:43, 962.68 examples/s]41168 examples [00:43, 964.57 examples/s]41265 examples [00:43, 964.82 examples/s]41362 examples [00:43, 950.90 examples/s]41459 examples [00:43, 955.46 examples/s]41555 examples [00:44, 948.95 examples/s]41651 examples [00:44, 951.82 examples/s]41752 examples [00:44, 965.26 examples/s]41849 examples [00:44, 935.17 examples/s]41947 examples [00:44, 946.96 examples/s]42046 examples [00:44, 957.92 examples/s]42142 examples [00:44, 956.22 examples/s]42239 examples [00:44, 958.25 examples/s]42336 examples [00:44, 958.79 examples/s]42435 examples [00:44, 965.71 examples/s]42534 examples [00:45, 971.03 examples/s]42635 examples [00:45, 980.88 examples/s]42735 examples [00:45, 985.53 examples/s]42834 examples [00:45, 969.10 examples/s]42934 examples [00:45, 977.72 examples/s]43032 examples [00:45, 977.54 examples/s]43130 examples [00:45, 948.07 examples/s]43226 examples [00:45, 912.21 examples/s]43318 examples [00:45, 874.61 examples/s]43411 examples [00:45, 888.27 examples/s]43511 examples [00:46, 916.78 examples/s]43604 examples [00:46, 905.27 examples/s]43700 examples [00:46, 920.99 examples/s]43798 examples [00:46, 935.06 examples/s]43897 examples [00:46, 948.46 examples/s]43995 examples [00:46, 956.19 examples/s]44096 examples [00:46, 969.21 examples/s]44194 examples [00:46, 941.50 examples/s]44290 examples [00:46, 943.86 examples/s]44385 examples [00:47, 928.40 examples/s]44479 examples [00:47, 913.97 examples/s]44571 examples [00:47, 898.31 examples/s]44665 examples [00:47, 910.16 examples/s]44764 examples [00:47, 930.84 examples/s]44858 examples [00:47, 930.48 examples/s]44952 examples [00:47, 910.54 examples/s]45046 examples [00:47, 918.78 examples/s]45142 examples [00:47, 928.58 examples/s]45236 examples [00:47, 929.17 examples/s]45333 examples [00:48, 940.81 examples/s]45430 examples [00:48, 947.62 examples/s]45530 examples [00:48, 961.26 examples/s]45627 examples [00:48, 960.77 examples/s]45726 examples [00:48, 967.27 examples/s]45826 examples [00:48, 973.99 examples/s]45924 examples [00:48, 975.17 examples/s]46022 examples [00:48, 976.29 examples/s]46120 examples [00:48, 950.11 examples/s]46218 examples [00:48, 958.01 examples/s]46317 examples [00:49, 965.95 examples/s]46416 examples [00:49, 971.44 examples/s]46514 examples [00:49, 960.76 examples/s]46611 examples [00:49, 954.90 examples/s]46710 examples [00:49, 963.09 examples/s]46809 examples [00:49, 968.60 examples/s]46908 examples [00:49, 974.52 examples/s]47006 examples [00:49, 976.15 examples/s]47104 examples [00:49, 974.42 examples/s]47202 examples [00:49, 968.37 examples/s]47299 examples [00:50, 967.64 examples/s]47396 examples [00:50, 967.72 examples/s]47496 examples [00:50, 976.86 examples/s]47594 examples [00:50, 961.05 examples/s]47691 examples [00:50, 943.88 examples/s]47786 examples [00:50, 940.55 examples/s]47889 examples [00:50, 964.54 examples/s]47991 examples [00:50, 977.47 examples/s]48089 examples [00:50, 971.30 examples/s]48187 examples [00:51, 945.77 examples/s]48282 examples [00:51, 931.59 examples/s]48379 examples [00:51, 940.08 examples/s]48477 examples [00:51, 949.57 examples/s]48573 examples [00:51, 949.01 examples/s]48672 examples [00:51, 958.30 examples/s]48768 examples [00:51, 954.95 examples/s]48864 examples [00:51, 932.57 examples/s]48962 examples [00:51, 945.25 examples/s]49058 examples [00:51, 948.59 examples/s]49153 examples [00:52, 897.41 examples/s]49252 examples [00:52, 922.43 examples/s]49347 examples [00:52, 930.05 examples/s]49449 examples [00:52, 955.08 examples/s]49545 examples [00:52, 945.48 examples/s]49640 examples [00:52, 943.79 examples/s]49738 examples [00:52, 952.63 examples/s]49836 examples [00:52, 959.17 examples/s]49935 examples [00:52, 965.86 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 14%|█▍        | 7068/50000 [00:00<00:00, 70673.50 examples/s] 37%|███▋      | 18326/50000 [00:00<00:00, 79556.35 examples/s] 60%|██████    | 30094/50000 [00:00<00:00, 88120.26 examples/s] 83%|████████▎ | 41565/50000 [00:00<00:00, 94705.52 examples/s]                                                               0 examples [00:00, ? examples/s]71 examples [00:00, 702.60 examples/s]145 examples [00:00, 711.98 examples/s]244 examples [00:00, 776.35 examples/s]350 examples [00:00, 842.89 examples/s]446 examples [00:00, 873.96 examples/s]546 examples [00:00, 907.30 examples/s]646 examples [00:00, 932.97 examples/s]745 examples [00:00, 948.79 examples/s]839 examples [00:00, 943.81 examples/s]940 examples [00:01, 961.52 examples/s]1037 examples [00:01, 963.44 examples/s]1135 examples [00:01, 965.45 examples/s]1236 examples [00:01, 975.72 examples/s]1337 examples [00:01, 983.61 examples/s]1437 examples [00:01, 987.05 examples/s]1536 examples [00:01, 962.34 examples/s]1633 examples [00:01, 952.88 examples/s]1729 examples [00:01, 945.98 examples/s]1827 examples [00:01, 955.30 examples/s]1924 examples [00:02, 958.68 examples/s]2020 examples [00:02, 945.93 examples/s]2117 examples [00:02, 952.49 examples/s]2215 examples [00:02, 960.07 examples/s]2314 examples [00:02, 966.64 examples/s]2414 examples [00:02, 975.10 examples/s]2512 examples [00:02, 966.81 examples/s]2609 examples [00:02, 945.73 examples/s]2704 examples [00:02, 939.46 examples/s]2803 examples [00:02, 953.07 examples/s]2901 examples [00:03, 960.30 examples/s]2998 examples [00:03, 955.16 examples/s]3096 examples [00:03, 961.38 examples/s]3193 examples [00:03, 961.03 examples/s]3290 examples [00:03, 947.07 examples/s]3390 examples [00:03, 961.87 examples/s]3487 examples [00:03, 956.41 examples/s]3586 examples [00:03, 964.60 examples/s]3683 examples [00:03, 964.05 examples/s]3780 examples [00:03, 960.49 examples/s]3877 examples [00:04, 944.61 examples/s]3972 examples [00:04, 929.11 examples/s]4072 examples [00:04, 947.35 examples/s]4168 examples [00:04, 948.05 examples/s]4266 examples [00:04, 957.22 examples/s]4364 examples [00:04, 963.89 examples/s]4461 examples [00:04, 932.77 examples/s]4562 examples [00:04, 954.57 examples/s]4659 examples [00:04, 957.00 examples/s]4758 examples [00:04, 964.48 examples/s]4862 examples [00:05, 983.59 examples/s]4961 examples [00:05, 974.48 examples/s]5059 examples [00:05, 971.73 examples/s]5157 examples [00:05, 963.34 examples/s]5256 examples [00:05, 969.32 examples/s]5354 examples [00:05, 972.38 examples/s]5452 examples [00:05, 957.84 examples/s]5549 examples [00:05, 958.69 examples/s]5648 examples [00:05, 965.93 examples/s]5749 examples [00:06, 978.71 examples/s]5850 examples [00:06, 984.61 examples/s]5949 examples [00:06, 948.42 examples/s]6045 examples [00:06, 934.24 examples/s]6139 examples [00:06, 935.54 examples/s]6233 examples [00:06, 909.33 examples/s]6325 examples [00:06, 896.29 examples/s]6420 examples [00:06, 909.71 examples/s]6520 examples [00:06, 934.14 examples/s]6621 examples [00:06, 953.91 examples/s]6717 examples [00:07, 953.26 examples/s]6813 examples [00:07, 951.69 examples/s]6909 examples [00:07, 950.77 examples/s]7008 examples [00:07, 961.42 examples/s]7107 examples [00:07, 969.56 examples/s]7208 examples [00:07, 980.51 examples/s]7307 examples [00:07, 971.97 examples/s]7405 examples [00:07, 956.12 examples/s]7501 examples [00:07, 949.36 examples/s]7597 examples [00:07, 920.70 examples/s]7693 examples [00:08, 931.59 examples/s]7787 examples [00:08, 897.76 examples/s]7880 examples [00:08, 905.36 examples/s]7979 examples [00:08, 926.74 examples/s]8078 examples [00:08, 943.17 examples/s]8177 examples [00:08, 954.57 examples/s]8273 examples [00:08, 940.47 examples/s]8372 examples [00:08, 952.47 examples/s]8474 examples [00:08, 970.11 examples/s]8574 examples [00:09, 978.01 examples/s]8673 examples [00:09, 979.85 examples/s]8772 examples [00:09, 964.53 examples/s]8872 examples [00:09, 973.29 examples/s]8971 examples [00:09, 975.69 examples/s]9069 examples [00:09, 965.10 examples/s]9168 examples [00:09, 969.81 examples/s]9266 examples [00:09, 948.82 examples/s]9367 examples [00:09, 964.35 examples/s]9468 examples [00:09, 974.64 examples/s]9568 examples [00:10, 981.15 examples/s]9669 examples [00:10, 989.01 examples/s]9768 examples [00:10, 975.25 examples/s]9866 examples [00:10, 975.49 examples/s]9966 examples [00:10, 981.53 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNX5FP7/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteNX5FP7/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['test', 'train'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1e75bca9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1e75bca9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1e75bca9d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1ee1059fd0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1e00c09898>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f1e75bca9d8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f1e75bca9d8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f1e75bca9d8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1e00c09748>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f1e545fd550>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f1dfa2f46a8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f1dfa2f46a8> 

  function with postional parmater data_info <function split_train_valid at 0x7f1dfa2f46a8> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f1dfa2f47b8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f1dfa2f47b8> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f1dfa2f47b8> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:00<19:48:31, 12.1kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:00<14:05:44, 17.0kB/s].vector_cache/glove.6B.zip:   0%|          | 221k/862M [00:00<9:55:12, 24.1kB/s]  .vector_cache/glove.6B.zip:   0%|          | 893k/862M [00:01<6:57:02, 34.4kB/s].vector_cache/glove.6B.zip:   0%|          | 3.46M/862M [00:01<4:51:13, 49.1kB/s].vector_cache/glove.6B.zip:   1%|          | 6.73M/862M [00:01<3:23:14, 70.2kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.3M/862M [00:01<2:21:24, 100kB/s] .vector_cache/glove.6B.zip:   2%|▏         | 15.9M/862M [00:01<1:38:42, 143kB/s].vector_cache/glove.6B.zip:   2%|▏         | 21.4M/862M [00:01<1:08:43, 204kB/s].vector_cache/glove.6B.zip:   3%|▎         | 24.6M/862M [00:01<48:03, 290kB/s]  .vector_cache/glove.6B.zip:   3%|▎         | 30.2M/862M [00:01<33:29, 414kB/s].vector_cache/glove.6B.zip:   4%|▍         | 33.1M/862M [00:01<23:29, 588kB/s].vector_cache/glove.6B.zip:   4%|▍         | 38.5M/862M [00:01<16:25, 836kB/s].vector_cache/glove.6B.zip:   5%|▍         | 41.7M/862M [00:02<11:34, 1.18MB/s].vector_cache/glove.6B.zip:   5%|▌         | 47.1M/862M [00:02<08:07, 1.67MB/s].vector_cache/glove.6B.zip:   6%|▌         | 50.2M/862M [00:02<05:47, 2.33MB/s].vector_cache/glove.6B.zip:   6%|▌         | 51.7M/862M [00:02<04:39, 2.90MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.8M/862M [00:04<05:09, 2.60MB/s].vector_cache/glove.6B.zip:   6%|▋         | 55.9M/862M [00:04<07:25, 1.81MB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.5M/862M [00:04<06:06, 2.20MB/s].vector_cache/glove.6B.zip:   7%|▋         | 58.8M/862M [00:04<04:29, 2.98MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:05<05:12, 2.57MB/s].vector_cache/glove.6B.zip:   7%|▋         | 59.6M/862M [00:06<8:30:40, 26.2kB/s].vector_cache/glove.6B.zip:   7%|▋         | 61.0M/862M [00:06<5:57:14, 37.4kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.7M/862M [00:08<4:11:52, 52.8kB/s].vector_cache/glove.6B.zip:   7%|▋         | 63.9M/862M [00:08<2:59:05, 74.3kB/s].vector_cache/glove.6B.zip:   7%|▋         | 64.7M/862M [00:08<2:05:49, 106kB/s] .vector_cache/glove.6B.zip:   8%|▊         | 66.9M/862M [00:08<1:28:00, 151kB/s].vector_cache/glove.6B.zip:   8%|▊         | 67.8M/862M [00:10<1:08:17, 194kB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.2M/862M [00:10<49:09, 269kB/s]  .vector_cache/glove.6B.zip:   8%|▊         | 69.8M/862M [00:10<34:40, 381kB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.9M/862M [00:12<27:18, 482kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.1M/862M [00:12<21:45, 605kB/s].vector_cache/glove.6B.zip:   8%|▊         | 72.9M/862M [00:12<15:50, 831kB/s].vector_cache/glove.6B.zip:   9%|▉         | 75.8M/862M [00:12<11:10, 1.17MB/s].vector_cache/glove.6B.zip:   9%|▉         | 76.0M/862M [00:14<37:29, 349kB/s] .vector_cache/glove.6B.zip:   9%|▉         | 76.4M/862M [00:14<27:20, 479kB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.5M/862M [00:14<19:27, 672kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.1M/862M [00:14<13:46, 947kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.2M/862M [00:16<1:20:22, 162kB/s].vector_cache/glove.6B.zip:   9%|▉         | 80.4M/862M [00:16<58:53, 221kB/s]  .vector_cache/glove.6B.zip:   9%|▉         | 81.1M/862M [00:16<41:51, 311kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.2M/862M [00:16<29:18, 442kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.3M/862M [00:18<1:36:14, 135kB/s].vector_cache/glove.6B.zip:  10%|▉         | 84.6M/862M [00:18<1:08:41, 189kB/s].vector_cache/glove.6B.zip:  10%|▉         | 86.1M/862M [00:18<48:20, 268kB/s]  .vector_cache/glove.6B.zip:  10%|█         | 88.4M/862M [00:20<36:37, 352kB/s].vector_cache/glove.6B.zip:  10%|█         | 88.6M/862M [00:20<28:26, 453kB/s].vector_cache/glove.6B.zip:  10%|█         | 89.4M/862M [00:20<20:35, 625kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.3M/862M [00:20<14:32, 883kB/s].vector_cache/glove.6B.zip:  11%|█         | 92.6M/862M [00:22<30:23, 422kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.0M/862M [00:22<22:36, 567kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.5M/862M [00:22<16:08, 793kB/s].vector_cache/glove.6B.zip:  11%|█         | 96.8M/862M [00:24<14:07, 904kB/s].vector_cache/glove.6B.zip:  11%|█         | 97.0M/862M [00:24<12:42, 1.00MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.7M/862M [00:24<09:27, 1.35MB/s].vector_cache/glove.6B.zip:  11%|█▏        | 99.0M/862M [00:24<06:53, 1.85MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<08:05, 1.57MB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 101M/862M [00:26<06:59, 1.81MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:26<05:10, 2.45MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<06:28, 1.95MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:28<07:10, 1.76MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:28<05:35, 2.26MB/s].vector_cache/glove.6B.zip:  12%|█▏        | 108M/862M [00:28<04:07, 3.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 109M/862M [00:29<06:49, 1.84MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:30<06:06, 2.05MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:30<04:33, 2.75MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 113M/862M [00:31<05:58, 2.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:32<05:19, 2.34MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:32<04:03, 3.06MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:33<05:35, 2.22MB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:34<05:14, 2.36MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:34<03:59, 3.10MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:35<05:32, 2.22MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 122M/862M [00:36<06:28, 1.90MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:36<05:05, 2.42MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 124M/862M [00:36<03:46, 3.26MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:37<06:34, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 126M/862M [00:38<05:56, 2.06MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:38<04:25, 2.77MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 130M/862M [00:39<05:49, 2.09MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:40<05:23, 2.26MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 132M/862M [00:40<04:05, 2.97MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 134M/862M [00:41<05:34, 2.18MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<06:26, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:42<05:09, 2.35MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 138M/862M [00:42<03:45, 3.21MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:43<21:39, 557kB/s] .vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:44<16:26, 733kB/s].vector_cache/glove.6B.zip:  16%|█▋        | 140M/862M [00:44<11:45, 1.02MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:45<10:54, 1.10MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:46<10:09, 1.18MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:46<07:46, 1.54MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:46<05:34, 2.14MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:47<22:10, 538kB/s] .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:48<16:50, 707kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 149M/862M [00:48<12:01, 989kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:49<11:03, 1.07MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:50<10:13, 1.16MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:50<07:47, 1.52MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:50<05:35, 2.11MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:51<21:59, 536kB/s] .vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:52<16:39, 707kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:52<11:56, 984kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 159M/862M [00:53<10:57, 1.07MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<10:15, 1.14MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:54<07:42, 1.52MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 162M/862M [00:54<05:34, 2.09MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<07:51, 1.48MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [00:55<06:43, 1.73MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [00:56<04:57, 2.35MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:03, 1.91MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [00:57<06:48, 1.70MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 169M/862M [00:58<05:18, 2.18MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 171M/862M [00:58<03:52, 2.98MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<07:34, 1.52MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [00:59<06:31, 1.76MB/s].vector_cache/glove.6B.zip:  20%|██        | 174M/862M [01:00<04:51, 2.36MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:57, 1.92MB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:01<05:23, 2.12MB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:02<04:01, 2.84MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<05:18, 2.14MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:03<06:06, 1.86MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:04<04:53, 2.32MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:04<03:33, 3.17MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:05<18:57, 596kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:05<14:29, 779kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:06<10:25, 1.08MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<09:47, 1.15MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:07<09:21, 1.20MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 190M/862M [01:08<07:09, 1.56MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:08<05:08, 2.17MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<21:08, 528kB/s] .vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:09<15:58, 698kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 195M/862M [01:10<11:27, 971kB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<10:28, 1.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:11<09:47, 1.13MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 198M/862M [01:12<07:19, 1.51MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:12<05:22, 2.06MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<06:40, 1.65MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:13<05:49, 1.89MB/s].vector_cache/glove.6B.zip:  24%|██▎       | 203M/862M [01:14<04:21, 2.52MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<05:30, 1.99MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:15<06:11, 1.77MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 206M/862M [01:15<04:48, 2.27MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 208M/862M [01:16<03:31, 3.09MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:17<06:29, 1.68MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 210M/862M [01:17<05:42, 1.90MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 211M/862M [01:17<04:16, 2.54MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<05:24, 2.00MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:19<06:03, 1.79MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 215M/862M [01:19<04:42, 2.29MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:20<03:26, 3.13MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<07:09, 1.50MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:21<05:56, 1.81MB/s].vector_cache/glove.6B.zip:  25%|██▌       | 220M/862M [01:21<04:26, 2.41MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<05:30, 1.94MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:23<04:57, 2.15MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:23<03:44, 2.84MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:25<05:03, 2.10MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 226M/862M [01:25<04:29, 2.36MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 228M/862M [01:25<03:25, 3.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<04:44, 2.22MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:27<05:38, 1.87MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:27<04:25, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 233M/862M [01:27<03:13, 3.26MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:29<07:53, 1.33MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 235M/862M [01:29<06:37, 1.58MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:29<04:51, 2.15MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<05:42, 1.82MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<06:16, 1.65MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:31<04:52, 2.13MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:31<03:31, 2.94MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<08:44, 1.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 243M/862M [01:33<07:00, 1.47MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 245M/862M [01:33<05:09, 1.99MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:58, 1.71MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 247M/862M [01:35<05:17, 1.94MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 249M/862M [01:35<03:55, 2.61MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:01, 2.03MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:37<05:39, 1.80MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 252M/862M [01:37<04:25, 2.30MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 253M/862M [01:37<03:17, 3.08MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:39<05:07, 1.97MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:39<04:40, 2.16MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 257M/862M [01:39<03:32, 2.85MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<04:42, 2.13MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:41<05:25, 1.85MB/s].vector_cache/glove.6B.zip:  30%|███       | 260M/862M [01:41<04:14, 2.37MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:41<03:07, 3.20MB/s].vector_cache/glove.6B.zip:  31%|███       | 263M/862M [01:43<05:32, 1.80MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:43<04:57, 2.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 265M/862M [01:43<03:43, 2.67MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:44<03:10, 3.13MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:45<7:01:57, 23.5kB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:45<4:55:35, 33.5kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:45<3:26:04, 47.8kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:47<2:31:53, 64.8kB/s].vector_cache/glove.6B.zip:  31%|███▏      | 272M/862M [01:47<1:48:27, 90.8kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 272M/862M [01:47<1:16:16, 129kB/s] .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:47<53:17, 184kB/s]  .vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:49<42:45, 229kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 276M/862M [01:49<30:55, 316kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:49<21:50, 446kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<17:28, 556kB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:51<14:21, 676kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 281M/862M [01:51<10:28, 926kB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:51<07:26, 1.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<09:05, 1.06MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 284M/862M [01:53<07:22, 1.31MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 286M/862M [01:53<05:24, 1.78MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<05:54, 1.62MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:55<06:14, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:55<04:48, 1.99MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [01:55<03:27, 2.75MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [01:57<09:04, 1.05MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 293M/862M [01:57<07:21, 1.29MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [01:57<05:20, 1.77MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<05:50, 1.61MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [01:59<06:10, 1.53MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [01:59<04:49, 1.95MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [01:59<03:29, 2.69MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:01<17:10, 545kB/s] .vector_cache/glove.6B.zip:  35%|███▍      | 301M/862M [02:01<13:00, 720kB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:01<09:19, 1.00MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:34, 1.08MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 305M/862M [02:03<08:02, 1.15MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:03<06:02, 1.54MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:03<04:21, 2.12MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<06:30, 1.42MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:05<05:21, 1.72MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 310M/862M [02:05<04:00, 2.30MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:05<02:55, 3.13MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<15:57, 574kB/s] .vector_cache/glove.6B.zip:  36%|███▋      | 313M/862M [02:07<13:00, 703kB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:07<09:29, 962kB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:07<06:44, 1.35MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<09:16, 979kB/s] .vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:09<07:17, 1.25MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:09<05:19, 1.70MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:44, 1.57MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 321M/862M [02:11<05:59, 1.50MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:11<04:36, 1.95MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:11<03:19, 2.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:13<07:27, 1.20MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:13<06:09, 1.45MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 327M/862M [02:13<04:29, 1.98MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:15<05:32, 1.60MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:15<03:58, 2.22MB/s].vector_cache/glove.6B.zip:  39%|███▊      | 334M/862M [02:16<05:17, 1.67MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:17<04:38, 1.90MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 336M/862M [02:17<03:26, 2.55MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:18<04:21, 2.00MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 338M/862M [02:19<04:53, 1.79MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 339M/862M [02:19<03:53, 2.24MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:19<02:48, 3.09MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:20<14:44, 588kB/s] .vector_cache/glove.6B.zip:  40%|███▉      | 342M/862M [02:21<11:14, 770kB/s].vector_cache/glove.6B.zip:  40%|███▉      | 344M/862M [02:21<08:04, 1.07MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:22<07:33, 1.14MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:23<07:10, 1.20MB/s].vector_cache/glove.6B.zip:  40%|████      | 347M/862M [02:23<05:24, 1.58MB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:23<03:53, 2.20MB/s].vector_cache/glove.6B.zip:  41%|████      | 350M/862M [02:24<06:51, 1.24MB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:25<05:41, 1.50MB/s].vector_cache/glove.6B.zip:  41%|████      | 352M/862M [02:25<04:11, 2.02MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:26<04:49, 1.75MB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:27<05:08, 1.64MB/s].vector_cache/glove.6B.zip:  41%|████      | 356M/862M [02:27<03:57, 2.13MB/s].vector_cache/glove.6B.zip:  41%|████▏     | 358M/862M [02:27<02:53, 2.91MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:28<05:29, 1.53MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 359M/862M [02:29<04:44, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:29<03:32, 2.36MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:30<04:19, 1.92MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:31<04:42, 1.77MB/s].vector_cache/glove.6B.zip:  42%|████▏     | 364M/862M [02:31<03:42, 2.24MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:31<02:40, 3.08MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<45:21, 182kB/s] .vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:32<32:35, 253kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:33<22:55, 359kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<17:50, 459kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 371M/862M [02:34<14:12, 576kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 372M/862M [02:35<10:17, 794kB/s].vector_cache/glove.6B.zip:  43%|████▎     | 375M/862M [02:35<07:15, 1.12MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:36<11:09, 727kB/s] .vector_cache/glove.6B.zip:  44%|████▎     | 376M/862M [02:36<08:40, 934kB/s].vector_cache/glove.6B.zip:  44%|████▎     | 377M/862M [02:37<06:16, 1.29MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 379M/862M [02:38<06:10, 1.30MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:38<06:00, 1.34MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:39<04:34, 1.76MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 383M/862M [02:39<03:17, 2.43MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 384M/862M [02:40<07:09, 1.11MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:40<05:50, 1.36MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:41<04:14, 1.87MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<04:47, 1.65MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:42<05:05, 1.55MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 389M/862M [02:42<03:55, 2.01MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 391M/862M [02:43<02:49, 2.78MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:44<07:31, 1.04MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 392M/862M [02:44<06:04, 1.29MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:44<04:24, 1.77MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<04:51, 1.60MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:46<05:01, 1.54MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 397M/862M [02:46<03:52, 2.00MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:47<02:47, 2.77MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 400M/862M [02:48<07:38, 1.01MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 401M/862M [02:48<06:09, 1.25MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:48<04:29, 1.71MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<04:53, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:50<05:06, 1.49MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 405M/862M [02:50<03:55, 1.94MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:50<02:50, 2.67MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 408M/862M [02:52<05:30, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 409M/862M [02:52<04:30, 1.67MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 410M/862M [02:52<03:20, 2.25MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 412M/862M [02:52<02:26, 3.07MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<20:21, 368kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<15:49, 473kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:54<11:24, 656kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [02:54<08:02, 925kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<09:17, 800kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 417M/862M [02:56<07:17, 1.02MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [02:56<05:16, 1.40MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:19, 1.38MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [02:58<05:16, 1.39MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 422M/862M [02:58<04:01, 1.83MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [02:58<02:52, 2.53MB/s].vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<07:28, 975kB/s] .vector_cache/glove.6B.zip:  49%|████▉     | 425M/862M [03:00<06:00, 1.21MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:00<04:23, 1.65MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:40, 1.55MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:02<04:47, 1.51MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 430M/862M [03:02<03:40, 1.96MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:02<02:39, 2.70MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:03<03:03, 2.34MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:04<4:48:08, 24.8kB/s].vector_cache/glove.6B.zip:  50%|█████     | 434M/862M [03:04<3:21:47, 35.4kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:04<2:20:26, 50.5kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:43:43, 68.3kB/s].vector_cache/glove.6B.zip:  51%|█████     | 437M/862M [03:06<1:14:04, 95.6kB/s].vector_cache/glove.6B.zip:  51%|█████     | 438M/862M [03:06<52:05, 136kB/s]   .vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:06<36:19, 193kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<30:16, 232kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:08<21:53, 320kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:08<15:25, 453kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:10<12:20, 563kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 446M/862M [03:10<09:21, 742kB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 447M/862M [03:10<06:42, 1.03MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 449M/862M [03:12<06:12, 1.11MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<05:51, 1.17MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:12<04:27, 1.54MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 453M/862M [03:12<03:11, 2.13MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<12:58, 525kB/s] .vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:14<09:46, 696kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 455M/862M [03:14<06:58, 973kB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<06:24, 1.05MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 458M/862M [03:16<05:10, 1.30MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:16<03:45, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<04:09, 1.60MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:18<03:36, 1.85MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:18<02:41, 2.47MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:23, 1.94MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:20<03:49, 1.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 467M/862M [03:20<03:01, 2.18MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 470M/862M [03:20<02:11, 2.98MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:21<11:50, 552kB/s] .vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:22<08:57, 729kB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:22<06:24, 1.01MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 474M/862M [03:23<05:57, 1.09MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 474M/862M [03:24<05:31, 1.17MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 475M/862M [03:24<04:11, 1.54MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:24<02:59, 2.13MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:25<21:23, 299kB/s] .vector_cache/glove.6B.zip:  56%|█████▌    | 479M/862M [03:26<15:38, 408kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 480M/862M [03:26<11:03, 576kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:27<09:07, 694kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<07:42, 821kB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 483M/862M [03:28<05:40, 1.11MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 486M/862M [03:28<04:01, 1.56MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:29<05:45, 1.09MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:30<04:41, 1.33MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:30<03:24, 1.83MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:47, 1.63MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:31<03:17, 1.88MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:32<02:27, 2.50MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:07, 1.96MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:33<03:31, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 496M/862M [03:34<02:44, 2.23MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:34<01:58, 3.06MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:57, 1.22MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 499M/862M [03:35<04:06, 1.47MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:36<03:01, 1.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:26, 1.74MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:37<03:39, 1.63MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 504M/862M [03:38<02:52, 2.07MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:38<02:05, 2.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:39<04:15, 1.39MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 508M/862M [03:39<03:36, 1.64MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:40<02:38, 2.23MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:09, 1.85MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:41<03:25, 1.70MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 512M/862M [03:42<02:39, 2.19MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 515M/862M [03:42<01:55, 3.00MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:56, 1.46MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 516M/862M [03:43<03:21, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:44<02:28, 2.32MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:02, 1.87MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:45<03:19, 1.71MB/s].vector_cache/glove.6B.zip:  60%|██████    | 521M/862M [03:45<02:34, 2.21MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:46<01:52, 3.02MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<04:06, 1.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 524M/862M [03:47<03:28, 1.62MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:47<02:33, 2.19MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:49<03:04, 1.82MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 528M/862M [03:49<02:42, 2.05MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:49<02:02, 2.72MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<02:41, 2.05MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 532M/862M [03:51<03:01, 1.81MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 533M/862M [03:51<02:22, 2.31MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 535M/862M [03:51<01:43, 3.16MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:53<03:44, 1.45MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 537M/862M [03:53<03:12, 1.69MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:53<02:22, 2.27MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:51, 1.88MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<03:06, 1.72MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 541M/862M [03:55<02:27, 2.17MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [03:55<01:46, 2.98MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<09:22, 564kB/s] .vector_cache/glove.6B.zip:  63%|██████▎   | 545M/862M [03:57<07:07, 742kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [03:57<05:06, 1.03MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:42, 1.11MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 549M/862M [03:59<04:21, 1.20MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [03:59<03:16, 1.59MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 552M/862M [03:59<02:20, 2.21MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<04:35, 1.12MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 553M/862M [04:01<03:44, 1.38MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:01<02:44, 1.87MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<03:04, 1.65MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:03<02:41, 1.88MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:03<02:00, 2.51MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:04<01:41, 2.98MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:05<3:34:13, 23.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 562M/862M [04:05<2:29:54, 33.4kB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:05<1:44:02, 47.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<1:16:33, 64.7kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 565M/862M [04:07<54:39, 90.6kB/s]  .vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:07<38:24, 129kB/s] .vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:07<26:40, 183kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<25:40, 190kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:09<18:23, 265kB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:09<12:56, 375kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:09<09:03, 532kB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 573M/862M [04:11<13:38, 353kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 573M/862M [04:11<10:33, 456kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 574M/862M [04:11<07:36, 631kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:11<05:21, 890kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:13<05:47, 818kB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 578M/862M [04:13<04:33, 1.04MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:13<03:17, 1.43MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<03:22, 1.39MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 582M/862M [04:15<02:51, 1.63MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:15<02:05, 2.22MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:29, 1.85MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 586M/862M [04:17<02:13, 2.07MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:17<01:39, 2.77MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:10, 2.09MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 590M/862M [04:19<02:28, 1.83MB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 591M/862M [04:19<01:56, 2.33MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:19<01:23, 3.21MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<04:20, 1.03MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 594M/862M [04:21<03:25, 1.30MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:21<02:28, 1.80MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:21<01:47, 2.45MB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<12:11, 361kB/s] .vector_cache/glove.6B.zip:  69%|██████▉   | 598M/862M [04:23<09:30, 463kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 599M/862M [04:23<06:50, 641kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:23<04:47, 907kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:25<07:28, 579kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 603M/862M [04:25<05:37, 769kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:25<04:01, 1.07MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<03:44, 1.14MB/s].vector_cache/glove.6B.zip:  70%|███████   | 607M/862M [04:27<02:59, 1.42MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:27<02:11, 1.93MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:30, 1.68MB/s].vector_cache/glove.6B.zip:  71%|███████   | 611M/862M [04:29<02:38, 1.59MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:29<02:01, 2.06MB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:29<01:27, 2.82MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:30<02:51, 1.44MB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 615M/862M [04:31<02:26, 1.69MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:31<01:47, 2.29MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:32<02:09, 1.88MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 619M/862M [04:33<02:24, 1.68MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:33<01:52, 2.16MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:33<01:20, 2.97MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:34<03:10, 1.25MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 623M/862M [04:35<02:35, 1.54MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:35<01:54, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 627M/862M [04:36<02:11, 1.78MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 628M/862M [04:37<01:57, 2.00MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:37<01:27, 2.65MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:38<01:52, 2.05MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<02:05, 1.83MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 632M/862M [04:39<01:39, 2.31MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 635M/862M [04:39<01:11, 3.16MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 636M/862M [04:40<27:51, 136kB/s] .vector_cache/glove.6B.zip:  74%|███████▍  | 636M/862M [04:41<19:51, 190kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:41<13:53, 269kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:42<10:29, 353kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 640M/862M [04:43<08:07, 456kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:43<05:49, 633kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 643M/862M [04:43<04:06, 892kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<04:17, 848kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 644M/862M [04:44<03:23, 1.07MB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 646M/862M [04:45<02:27, 1.47MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:29, 1.43MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 648M/862M [04:46<02:31, 1.41MB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:47<01:56, 1.84MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:47<01:22, 2.55MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 652M/862M [04:48<04:43, 741kB/s] .vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:48<03:39, 953kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:49<02:38, 1.32MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:37, 1.31MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 656M/862M [04:50<02:33, 1.34MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 657M/862M [04:51<01:57, 1.75MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:51<01:23, 2.43MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 660M/862M [04:52<04:42, 715kB/s] .vector_cache/glove.6B.zip:  77%|███████▋  | 661M/862M [04:52<03:39, 919kB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:53<02:37, 1.27MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:33, 1.29MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 665M/862M [04:54<02:28, 1.33MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [04:55<01:54, 1.72MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 668M/862M [04:55<01:21, 2.38MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<05:34, 578kB/s] .vector_cache/glove.6B.zip:  78%|███████▊  | 669M/862M [04:56<04:14, 758kB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 671M/862M [04:57<03:02, 1.05MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:48, 1.12MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 673M/862M [04:58<02:37, 1.20MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [04:59<02:00, 1.57MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [04:59<01:25, 2.17MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 677M/862M [05:00<05:43, 538kB/s] .vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:00<04:16, 719kB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 679M/862M [05:00<03:04, 997kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:01<02:09, 1.39MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 681M/862M [05:02<05:24, 558kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:02<04:05, 736kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:03<02:54, 1.02MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 685M/862M [05:04<02:41, 1.09MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:04<02:11, 1.34MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:04<01:36, 1.82MB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 690M/862M [05:06<01:45, 1.64MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:06<01:49, 1.57MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:06<01:24, 2.02MB/s].vector_cache/glove.6B.zip:  80%|████████  | 693M/862M [05:07<01:00, 2.78MB/s].vector_cache/glove.6B.zip:  80%|████████  | 694M/862M [05:08<04:27, 629kB/s] .vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:08<03:24, 820kB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:08<02:26, 1.14MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<02:19, 1.18MB/s].vector_cache/glove.6B.zip:  81%|████████  | 698M/862M [05:10<01:54, 1.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 700M/862M [05:10<01:23, 1.95MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:33, 1.71MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 702M/862M [05:12<01:37, 1.63MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:12<01:16, 2.09MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:13<00:54, 2.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 706M/862M [05:14<19:14, 135kB/s] .vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:14<13:42, 189kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:14<09:34, 268kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 710M/862M [05:16<07:10, 353kB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:16<05:15, 479kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 712M/862M [05:16<03:42, 675kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 714M/862M [05:18<03:07, 786kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<02:42, 907kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:18<01:59, 1.22MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 718M/862M [05:18<01:24, 1.71MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<02:20, 1.02MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:20<01:53, 1.26MB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 720M/862M [05:20<01:22, 1.72MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:27, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:22<01:32, 1.51MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:22<01:11, 1.93MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:22<00:50, 2.66MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<04:08, 544kB/s] .vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:24<03:07, 718kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:24<02:13, 1.00MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:25<01:39, 1.32MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<1:34:46, 23.1kB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 731M/862M [05:26<1:06:06, 33.0kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:26<45:33, 47.1kB/s]  .vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<32:24, 65.5kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 735M/862M [05:28<23:07, 91.7kB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:28<16:11, 130kB/s] .vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:28<11:09, 186kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<08:42, 236kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 739M/862M [05:30<06:15, 327kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 741M/862M [05:30<04:23, 461kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<03:27, 574kB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 743M/862M [05:32<02:49, 700kB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:32<02:03, 954kB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:32<01:26, 1.34MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 747M/862M [05:34<01:45, 1.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:34<01:25, 1.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 749M/862M [05:34<01:01, 1.83MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 751M/862M [05:36<01:07, 1.64MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<01:09, 1.60MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:36<00:52, 2.07MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 754M/862M [05:36<00:38, 2.83MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:11, 1.48MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:38<01:01, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:38<00:45, 2.32MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:53, 1.90MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:40<00:59, 1.73MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:40<00:46, 2.19MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:40<00:32, 3.01MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<12:07, 135kB/s] .vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:42<08:37, 189kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 766M/862M [05:42<05:59, 268kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<04:26, 353kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 768M/862M [05:44<03:25, 456kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:44<02:27, 632kB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 771M/862M [05:44<01:41, 894kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 772M/862M [05:46<02:11, 686kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:46<01:40, 888kB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:46<01:11, 1.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 776M/862M [05:48<01:08, 1.25MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<01:05, 1.30MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:48<00:49, 1.72MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:48<00:34, 2.37MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 780M/862M [05:50<01:04, 1.26MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:50<00:53, 1.52MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 782M/862M [05:50<00:38, 2.07MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:51<00:44, 1.76MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:52<00:37, 2.06MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:52<00:28, 2.69MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 788M/862M [05:52<00:20, 3.64MB/s].vector_cache/glove.6B.zip:  91%|█████████▏| 789M/862M [05:53<02:02, 598kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:54<01:42, 717kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 790M/862M [05:54<01:14, 976kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 792M/862M [05:54<00:50, 1.37MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:55<01:31, 757kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [05:56<01:09, 989kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 795M/862M [05:56<00:49, 1.36MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:57<00:48, 1.34MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [05:58<00:48, 1.35MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [05:58<00:36, 1.74MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:58<00:25, 2.41MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [05:59<01:53, 536kB/s] .vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:00<01:25, 709kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:00<00:59, 991kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:01<00:53, 1.07MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 805M/862M [06:02<00:49, 1.15MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 806M/862M [06:02<00:36, 1.51MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:02<00:25, 2.11MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:03<01:37, 543kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:04<01:13, 716kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:04<00:51, 996kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:05<00:44, 1.08MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:06<00:41, 1.16MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 815M/862M [06:06<00:30, 1.54MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 816M/862M [06:06<00:22, 2.09MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:25, 1.73MB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:07<00:22, 1.95MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:08<00:16, 2.60MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:20, 2.00MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:09<00:22, 1.76MB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:10<00:17, 2.21MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:10<00:12, 3.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<01:05, 553kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:11<00:48, 738kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:12<00:33, 1.03MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:12<00:22, 1.44MB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:35, 335kB/s] .vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:13<01:13, 434kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:14<00:51, 601kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:14<00:33, 850kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:15<01:27, 318kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:15<01:03, 433kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:16<00:42, 609kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:32, 728kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:27, 852kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:17<00:19, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:18<00:12, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:17, 1.14MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:19<00:13, 1.38MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:19<00:09, 1.88MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.67MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:21<00:09, 1.61MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:21<00:06, 2.09MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:22<00:04, 2.86MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:07, 1.52MB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:23<00:06, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 853M/862M [06:23<00:03, 2.36MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.92MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:25<00:03, 1.76MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:25<00:02, 2.23MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:26<00:00, 3.07MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:27<00:16, 183kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:27<00:10, 254kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:27<00:02, 360kB/s].vector_cache/glove.6B.zip: 862MB [06:27, 2.22MB/s]                          
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 1/400000 [00:00<63:33:14,  1.75it/s]  0%|          | 807/400000 [00:00<44:24:08,  2.50it/s]  0%|          | 1550/400000 [00:00<31:01:41,  3.57it/s]  1%|          | 2307/400000 [00:00<21:40:58,  5.09it/s]  1%|          | 3029/400000 [00:00<15:09:18,  7.28it/s]  1%|          | 3819/400000 [00:01<10:35:29, 10.39it/s]  1%|          | 4611/400000 [00:01<7:24:12, 14.83it/s]   1%|▏         | 5369/400000 [00:01<5:10:36, 21.18it/s]  2%|▏         | 6135/400000 [00:01<3:37:15, 30.21it/s]  2%|▏         | 6868/400000 [00:01<2:32:04, 43.09it/s]  2%|▏         | 7677/400000 [00:01<1:46:28, 61.41it/s]  2%|▏         | 8474/400000 [00:01<1:14:37, 87.44it/s]  2%|▏         | 9288/400000 [00:01<52:22, 124.35it/s]   3%|▎         | 10077/400000 [00:01<36:49, 176.45it/s]  3%|▎         | 10860/400000 [00:01<25:59, 249.57it/s]  3%|▎         | 11634/400000 [00:02<18:25, 351.28it/s]  3%|▎         | 12423/400000 [00:02<13:07, 492.42it/s]  3%|▎         | 13205/400000 [00:02<09:24, 684.97it/s]  3%|▎         | 13976/400000 [00:02<06:49, 942.03it/s]  4%|▎         | 14758/400000 [00:02<05:01, 1279.62it/s]  4%|▍         | 15547/400000 [00:02<03:44, 1709.17it/s]  4%|▍         | 16324/400000 [00:02<02:51, 2231.23it/s]  4%|▍         | 17127/400000 [00:02<02:14, 2848.23it/s]  4%|▍         | 17912/400000 [00:02<01:48, 3517.42it/s]  5%|▍         | 18695/400000 [00:02<01:32, 4139.49it/s]  5%|▍         | 19479/400000 [00:03<01:18, 4822.24it/s]  5%|▌         | 20253/400000 [00:03<01:09, 5435.51it/s]  5%|▌         | 21026/400000 [00:03<01:03, 5966.78it/s]  5%|▌         | 21797/400000 [00:03<00:59, 6398.57it/s]  6%|▌         | 22567/400000 [00:03<00:56, 6700.48it/s]  6%|▌         | 23348/400000 [00:03<00:53, 6996.92it/s]  6%|▌         | 24132/400000 [00:03<00:51, 7229.98it/s]  6%|▌         | 24934/400000 [00:03<00:50, 7448.83it/s]  6%|▋         | 25747/400000 [00:03<00:49, 7636.62it/s]  7%|▋         | 26538/400000 [00:04<00:49, 7595.14it/s]  7%|▋         | 27317/400000 [00:04<00:49, 7595.97it/s]  7%|▋         | 28090/400000 [00:04<00:49, 7456.12it/s]  7%|▋         | 28864/400000 [00:04<00:49, 7538.82it/s]  7%|▋         | 29642/400000 [00:04<00:48, 7606.82it/s]  8%|▊         | 30436/400000 [00:04<00:47, 7701.72it/s]  8%|▊         | 31230/400000 [00:04<00:47, 7770.59it/s]  8%|▊         | 32010/400000 [00:04<00:47, 7728.24it/s]  8%|▊         | 32813/400000 [00:04<00:46, 7814.35it/s]  8%|▊         | 33639/400000 [00:04<00:46, 7942.27it/s]  9%|▊         | 34444/400000 [00:05<00:45, 7971.99it/s]  9%|▉         | 35274/400000 [00:05<00:45, 8065.52it/s]  9%|▉         | 36139/400000 [00:05<00:44, 8231.49it/s]  9%|▉         | 37009/400000 [00:05<00:43, 8364.78it/s]  9%|▉         | 37848/400000 [00:05<00:43, 8355.53it/s] 10%|▉         | 38685/400000 [00:05<00:44, 8143.38it/s] 10%|▉         | 39506/400000 [00:05<00:44, 8161.56it/s] 10%|█         | 40324/400000 [00:05<00:44, 8051.97it/s] 10%|█         | 41131/400000 [00:05<00:44, 8030.90it/s] 10%|█         | 41936/400000 [00:05<00:45, 7851.69it/s] 11%|█         | 42723/400000 [00:06<00:45, 7842.10it/s] 11%|█         | 43595/400000 [00:06<00:44, 8083.95it/s] 11%|█         | 44422/400000 [00:06<00:43, 8137.72it/s] 11%|█▏        | 45275/400000 [00:06<00:43, 8249.04it/s] 12%|█▏        | 46102/400000 [00:06<00:43, 8125.79it/s] 12%|█▏        | 46917/400000 [00:06<00:44, 8009.33it/s] 12%|█▏        | 47741/400000 [00:06<00:43, 8076.30it/s] 12%|█▏        | 48550/400000 [00:06<00:43, 8007.89it/s] 12%|█▏        | 49359/400000 [00:06<00:43, 8029.68it/s] 13%|█▎        | 50163/400000 [00:06<00:43, 8028.71it/s] 13%|█▎        | 50985/400000 [00:07<00:43, 8083.16it/s] 13%|█▎        | 51821/400000 [00:07<00:42, 8163.80it/s] 13%|█▎        | 52638/400000 [00:07<00:42, 8139.94it/s] 13%|█▎        | 53453/400000 [00:07<00:43, 7908.04it/s] 14%|█▎        | 54246/400000 [00:07<00:44, 7847.26it/s] 14%|█▍        | 55033/400000 [00:07<00:44, 7810.02it/s] 14%|█▍        | 55836/400000 [00:07<00:43, 7874.54it/s] 14%|█▍        | 56632/400000 [00:07<00:43, 7899.57it/s] 14%|█▍        | 57423/400000 [00:07<00:43, 7869.88it/s] 15%|█▍        | 58228/400000 [00:07<00:43, 7921.34it/s] 15%|█▍        | 59059/400000 [00:08<00:42, 8032.09it/s] 15%|█▍        | 59894/400000 [00:08<00:41, 8122.84it/s] 15%|█▌        | 60707/400000 [00:08<00:41, 8108.84it/s] 15%|█▌        | 61543/400000 [00:08<00:41, 8180.59it/s] 16%|█▌        | 62380/400000 [00:08<00:40, 8234.83it/s] 16%|█▌        | 63204/400000 [00:08<00:41, 8131.57it/s] 16%|█▌        | 64031/400000 [00:08<00:41, 8169.74it/s] 16%|█▌        | 64862/400000 [00:08<00:40, 8208.35it/s] 16%|█▋        | 65684/400000 [00:08<00:41, 8071.16it/s] 17%|█▋        | 66492/400000 [00:08<00:41, 7991.52it/s] 17%|█▋        | 67295/400000 [00:09<00:41, 7999.95it/s] 17%|█▋        | 68096/400000 [00:09<00:41, 7961.98it/s] 17%|█▋        | 68893/400000 [00:09<00:42, 7880.19it/s] 17%|█▋        | 69685/400000 [00:09<00:41, 7891.64it/s] 18%|█▊        | 70475/400000 [00:09<00:42, 7843.18it/s] 18%|█▊        | 71260/400000 [00:09<00:42, 7788.38it/s] 18%|█▊        | 72063/400000 [00:09<00:41, 7856.81it/s] 18%|█▊        | 72850/400000 [00:09<00:41, 7840.50it/s] 18%|█▊        | 73685/400000 [00:09<00:40, 7985.71it/s] 19%|█▊        | 74497/400000 [00:10<00:40, 8023.54it/s] 19%|█▉        | 75300/400000 [00:10<00:40, 7983.42it/s] 19%|█▉        | 76135/400000 [00:10<00:40, 8087.49it/s] 19%|█▉        | 76957/400000 [00:10<00:39, 8126.47it/s] 19%|█▉        | 77791/400000 [00:10<00:39, 8188.55it/s] 20%|█▉        | 78611/400000 [00:10<00:39, 8083.32it/s] 20%|█▉        | 79420/400000 [00:10<00:40, 7927.90it/s] 20%|██        | 80233/400000 [00:10<00:40, 7985.99it/s] 20%|██        | 81036/400000 [00:10<00:39, 7997.21it/s] 20%|██        | 81894/400000 [00:10<00:38, 8161.45it/s] 21%|██        | 82712/400000 [00:11<00:39, 8128.87it/s] 21%|██        | 83526/400000 [00:11<00:40, 7747.25it/s] 21%|██        | 84346/400000 [00:11<00:40, 7875.07it/s] 21%|██▏       | 85210/400000 [00:11<00:38, 8087.31it/s] 22%|██▏       | 86023/400000 [00:11<00:40, 7814.26it/s] 22%|██▏       | 86810/400000 [00:11<00:41, 7499.66it/s] 22%|██▏       | 87589/400000 [00:11<00:41, 7582.10it/s] 22%|██▏       | 88405/400000 [00:11<00:40, 7745.58it/s] 22%|██▏       | 89256/400000 [00:11<00:39, 7959.54it/s] 23%|██▎       | 90057/400000 [00:11<00:39, 7913.93it/s] 23%|██▎       | 90852/400000 [00:12<00:39, 7812.95it/s] 23%|██▎       | 91636/400000 [00:12<00:39, 7726.50it/s] 23%|██▎       | 92469/400000 [00:12<00:38, 7897.11it/s] 23%|██▎       | 93290/400000 [00:12<00:38, 7985.78it/s] 24%|██▎       | 94091/400000 [00:12<00:38, 7939.07it/s] 24%|██▎       | 94901/400000 [00:12<00:38, 7984.35it/s] 24%|██▍       | 95701/400000 [00:12<00:38, 7950.02it/s] 24%|██▍       | 96497/400000 [00:12<00:38, 7800.01it/s] 24%|██▍       | 97282/400000 [00:12<00:38, 7813.93it/s] 25%|██▍       | 98072/400000 [00:12<00:38, 7837.55it/s] 25%|██▍       | 98862/400000 [00:13<00:38, 7853.00it/s] 25%|██▍       | 99650/400000 [00:13<00:38, 7860.31it/s] 25%|██▌       | 100437/400000 [00:13<00:38, 7818.95it/s] 25%|██▌       | 101234/400000 [00:13<00:38, 7859.87it/s] 26%|██▌       | 102021/400000 [00:13<00:38, 7783.00it/s] 26%|██▌       | 102800/400000 [00:13<00:38, 7647.63it/s] 26%|██▌       | 103566/400000 [00:13<00:39, 7545.84it/s] 26%|██▌       | 104351/400000 [00:13<00:38, 7632.80it/s] 26%|██▋       | 105175/400000 [00:13<00:37, 7804.97it/s] 26%|██▋       | 105986/400000 [00:13<00:37, 7892.44it/s] 27%|██▋       | 106793/400000 [00:14<00:36, 7944.58it/s] 27%|██▋       | 107622/400000 [00:14<00:36, 8043.08it/s] 27%|██▋       | 108471/400000 [00:14<00:35, 8170.38it/s] 27%|██▋       | 109297/400000 [00:14<00:35, 8196.52it/s] 28%|██▊       | 110118/400000 [00:14<00:36, 8014.10it/s] 28%|██▊       | 110921/400000 [00:14<00:36, 7955.49it/s] 28%|██▊       | 111718/400000 [00:14<00:36, 7845.41it/s] 28%|██▊       | 112504/400000 [00:14<00:36, 7800.96it/s] 28%|██▊       | 113291/400000 [00:14<00:36, 7821.15it/s] 29%|██▊       | 114099/400000 [00:15<00:36, 7896.15it/s] 29%|██▊       | 114925/400000 [00:15<00:35, 7999.87it/s] 29%|██▉       | 115726/400000 [00:15<00:35, 7939.65it/s] 29%|██▉       | 116542/400000 [00:15<00:35, 8001.89it/s] 29%|██▉       | 117347/400000 [00:15<00:35, 8012.61it/s] 30%|██▉       | 118149/400000 [00:15<00:36, 7794.70it/s] 30%|██▉       | 118959/400000 [00:15<00:35, 7881.20it/s] 30%|██▉       | 119756/400000 [00:15<00:35, 7906.65it/s] 30%|███       | 120595/400000 [00:15<00:34, 8043.45it/s] 30%|███       | 121401/400000 [00:15<00:34, 8022.89it/s] 31%|███       | 122224/400000 [00:16<00:34, 8083.73it/s] 31%|███       | 123058/400000 [00:16<00:33, 8157.52it/s] 31%|███       | 123875/400000 [00:16<00:34, 8047.07it/s] 31%|███       | 124684/400000 [00:16<00:34, 8058.48it/s] 31%|███▏      | 125491/400000 [00:16<00:34, 7938.70it/s] 32%|███▏      | 126286/400000 [00:16<00:36, 7475.19it/s] 32%|███▏      | 127126/400000 [00:16<00:35, 7728.15it/s] 32%|███▏      | 127918/400000 [00:16<00:34, 7782.83it/s] 32%|███▏      | 128727/400000 [00:16<00:34, 7872.43it/s] 32%|███▏      | 129540/400000 [00:16<00:34, 7945.85it/s] 33%|███▎      | 130396/400000 [00:17<00:33, 8119.98it/s] 33%|███▎      | 131245/400000 [00:17<00:32, 8225.75it/s] 33%|███▎      | 132070/400000 [00:17<00:32, 8216.37it/s] 33%|███▎      | 132947/400000 [00:17<00:31, 8373.88it/s] 33%|███▎      | 133796/400000 [00:17<00:31, 8407.31it/s] 34%|███▎      | 134639/400000 [00:17<00:32, 8230.00it/s] 34%|███▍      | 135464/400000 [00:17<00:32, 8046.79it/s] 34%|███▍      | 136271/400000 [00:17<00:32, 8036.03it/s] 34%|███▍      | 137077/400000 [00:17<00:32, 8039.38it/s] 34%|███▍      | 137902/400000 [00:17<00:32, 8099.88it/s] 35%|███▍      | 138771/400000 [00:18<00:31, 8265.96it/s] 35%|███▍      | 139612/400000 [00:18<00:31, 8308.21it/s] 35%|███▌      | 140444/400000 [00:18<00:31, 8120.38it/s] 35%|███▌      | 141265/400000 [00:18<00:31, 8144.98it/s] 36%|███▌      | 142083/400000 [00:18<00:31, 8152.68it/s] 36%|███▌      | 142900/400000 [00:18<00:31, 8152.70it/s] 36%|███▌      | 143716/400000 [00:18<00:31, 8152.52it/s] 36%|███▌      | 144532/400000 [00:18<00:31, 8051.36it/s] 36%|███▋      | 145369/400000 [00:18<00:31, 8142.88it/s] 37%|███▋      | 146184/400000 [00:18<00:31, 8131.13it/s] 37%|███▋      | 147000/400000 [00:19<00:31, 8137.66it/s] 37%|███▋      | 147815/400000 [00:19<00:31, 8003.51it/s] 37%|███▋      | 148627/400000 [00:19<00:31, 8036.23it/s] 37%|███▋      | 149453/400000 [00:19<00:30, 8099.94it/s] 38%|███▊      | 150264/400000 [00:19<00:31, 8018.55it/s] 38%|███▊      | 151127/400000 [00:19<00:30, 8192.51it/s] 38%|███▊      | 151986/400000 [00:19<00:29, 8305.73it/s] 38%|███▊      | 152818/400000 [00:19<00:30, 8181.03it/s] 38%|███▊      | 153662/400000 [00:19<00:29, 8254.84it/s] 39%|███▊      | 154509/400000 [00:20<00:29, 8318.04it/s] 39%|███▉      | 155342/400000 [00:20<00:29, 8318.64it/s] 39%|███▉      | 156175/400000 [00:20<00:29, 8249.41it/s] 39%|███▉      | 157001/400000 [00:20<00:29, 8219.53it/s] 39%|███▉      | 157846/400000 [00:20<00:29, 8286.06it/s] 40%|███▉      | 158676/400000 [00:20<00:29, 8222.01it/s] 40%|███▉      | 159500/400000 [00:20<00:29, 8224.10it/s] 40%|████      | 160323/400000 [00:20<00:29, 8176.36it/s] 40%|████      | 161141/400000 [00:20<00:29, 8074.55it/s] 40%|████      | 161976/400000 [00:20<00:29, 8154.02it/s] 41%|████      | 162801/400000 [00:21<00:28, 8182.33it/s] 41%|████      | 163620/400000 [00:21<00:29, 8095.64it/s] 41%|████      | 164431/400000 [00:21<00:29, 7950.65it/s] 41%|████▏     | 165227/400000 [00:21<00:29, 7926.35it/s] 42%|████▏     | 166021/400000 [00:21<00:29, 7912.42it/s] 42%|████▏     | 166850/400000 [00:21<00:29, 8020.10it/s] 42%|████▏     | 167721/400000 [00:21<00:28, 8213.86it/s] 42%|████▏     | 168545/400000 [00:21<00:28, 8201.23it/s] 42%|████▏     | 169367/400000 [00:21<00:28, 8023.55it/s] 43%|████▎     | 170246/400000 [00:21<00:27, 8238.76it/s] 43%|████▎     | 171088/400000 [00:22<00:27, 8291.31it/s] 43%|████▎     | 171970/400000 [00:22<00:27, 8441.53it/s] 43%|████▎     | 172817/400000 [00:22<00:27, 8403.35it/s] 43%|████▎     | 173659/400000 [00:22<00:27, 8086.88it/s] 44%|████▎     | 174472/400000 [00:22<00:27, 8094.53it/s] 44%|████▍     | 175284/400000 [00:22<00:27, 8094.06it/s] 44%|████▍     | 176149/400000 [00:22<00:27, 8252.78it/s] 44%|████▍     | 177000/400000 [00:22<00:26, 8326.20it/s] 44%|████▍     | 177835/400000 [00:22<00:27, 8123.42it/s] 45%|████▍     | 178671/400000 [00:22<00:27, 8192.19it/s] 45%|████▍     | 179493/400000 [00:23<00:26, 8198.18it/s] 45%|████▌     | 180314/400000 [00:23<00:26, 8163.47it/s] 45%|████▌     | 181132/400000 [00:23<00:26, 8121.67it/s] 45%|████▌     | 181945/400000 [00:23<00:27, 7932.33it/s] 46%|████▌     | 182740/400000 [00:23<00:27, 7898.79it/s] 46%|████▌     | 183559/400000 [00:23<00:27, 7962.10it/s] 46%|████▌     | 184357/400000 [00:23<00:27, 7902.53it/s] 46%|████▋     | 185148/400000 [00:23<00:27, 7796.77it/s] 46%|████▋     | 185944/400000 [00:23<00:27, 7841.92it/s] 47%|████▋     | 186739/400000 [00:23<00:27, 7873.14it/s] 47%|████▋     | 187527/400000 [00:24<00:27, 7865.35it/s] 47%|████▋     | 188333/400000 [00:24<00:26, 7922.55it/s] 47%|████▋     | 189126/400000 [00:24<00:27, 7723.54it/s] 47%|████▋     | 189900/400000 [00:24<00:27, 7693.72it/s] 48%|████▊     | 190671/400000 [00:24<00:27, 7652.03it/s] 48%|████▊     | 191484/400000 [00:24<00:26, 7787.85it/s] 48%|████▊     | 192283/400000 [00:24<00:26, 7847.23it/s] 48%|████▊     | 193069/400000 [00:24<00:26, 7777.26it/s] 48%|████▊     | 193848/400000 [00:24<00:26, 7681.92it/s] 49%|████▊     | 194617/400000 [00:25<00:27, 7605.15it/s] 49%|████▉     | 195475/400000 [00:25<00:25, 7871.47it/s] 49%|████▉     | 196277/400000 [00:25<00:25, 7915.32it/s] 49%|████▉     | 197087/400000 [00:25<00:25, 7966.85it/s] 49%|████▉     | 197886/400000 [00:25<00:25, 7893.41it/s] 50%|████▉     | 198677/400000 [00:25<00:25, 7793.44it/s] 50%|████▉     | 199504/400000 [00:25<00:25, 7928.73it/s] 50%|█████     | 200347/400000 [00:25<00:24, 8070.13it/s] 50%|█████     | 201174/400000 [00:25<00:24, 8127.48it/s] 50%|█████     | 201988/400000 [00:25<00:24, 8047.86it/s] 51%|█████     | 202808/400000 [00:26<00:24, 8090.93it/s] 51%|█████     | 203618/400000 [00:26<00:24, 8068.80it/s] 51%|█████     | 204437/400000 [00:26<00:24, 8103.57it/s] 51%|█████▏    | 205249/400000 [00:26<00:24, 8105.94it/s] 52%|█████▏    | 206060/400000 [00:26<00:24, 7927.20it/s] 52%|█████▏    | 206854/400000 [00:26<00:24, 7833.13it/s] 52%|█████▏    | 207639/400000 [00:26<00:25, 7653.61it/s] 52%|█████▏    | 208434/400000 [00:26<00:24, 7737.63it/s] 52%|█████▏    | 209232/400000 [00:26<00:24, 7806.17it/s] 53%|█████▎    | 210014/400000 [00:26<00:24, 7722.16it/s] 53%|█████▎    | 210825/400000 [00:27<00:24, 7834.04it/s] 53%|█████▎    | 211679/400000 [00:27<00:23, 8032.84it/s] 53%|█████▎    | 212492/400000 [00:27<00:23, 8060.65it/s] 53%|█████▎    | 213354/400000 [00:27<00:22, 8217.03it/s] 54%|█████▎    | 214178/400000 [00:27<00:23, 8072.62it/s] 54%|█████▍    | 215030/400000 [00:27<00:22, 8201.77it/s] 54%|█████▍    | 215852/400000 [00:27<00:22, 8157.01it/s] 54%|█████▍    | 216669/400000 [00:27<00:22, 7980.10it/s] 54%|█████▍    | 217469/400000 [00:27<00:23, 7911.72it/s] 55%|█████▍    | 218299/400000 [00:27<00:22, 8022.83it/s] 55%|█████▍    | 219140/400000 [00:28<00:22, 8131.14it/s] 55%|█████▍    | 219955/400000 [00:28<00:22, 8101.46it/s] 55%|█████▌    | 220793/400000 [00:28<00:21, 8181.33it/s] 55%|█████▌    | 221621/400000 [00:28<00:21, 8208.64it/s] 56%|█████▌    | 222443/400000 [00:28<00:21, 8102.03it/s] 56%|█████▌    | 223254/400000 [00:28<00:21, 8079.03it/s] 56%|█████▌    | 224084/400000 [00:28<00:21, 8142.13it/s] 56%|█████▌    | 224900/400000 [00:28<00:21, 8144.41it/s] 56%|█████▋    | 225729/400000 [00:28<00:21, 8186.94it/s] 57%|█████▋    | 226548/400000 [00:28<00:21, 8185.53it/s] 57%|█████▋    | 227382/400000 [00:29<00:20, 8231.24it/s] 57%|█████▋    | 228221/400000 [00:29<00:20, 8276.29it/s] 57%|█████▋    | 229050/400000 [00:29<00:20, 8279.39it/s] 57%|█████▋    | 229910/400000 [00:29<00:20, 8370.30it/s] 58%|█████▊    | 230748/400000 [00:29<00:20, 8243.08it/s] 58%|█████▊    | 231617/400000 [00:29<00:20, 8371.99it/s] 58%|█████▊    | 232456/400000 [00:29<00:20, 8365.79it/s] 58%|█████▊    | 233294/400000 [00:29<00:20, 8066.66it/s] 59%|█████▊    | 234109/400000 [00:29<00:20, 8090.14it/s] 59%|█████▊    | 234922/400000 [00:29<00:20, 8100.09it/s] 59%|█████▉    | 235744/400000 [00:30<00:20, 8132.32it/s] 59%|█████▉    | 236619/400000 [00:30<00:19, 8307.13it/s] 59%|█████▉    | 237452/400000 [00:30<00:19, 8298.11it/s] 60%|█████▉    | 238283/400000 [00:30<00:19, 8200.83it/s] 60%|█████▉    | 239105/400000 [00:30<00:19, 8071.99it/s] 60%|█████▉    | 239914/400000 [00:30<00:19, 8059.39it/s] 60%|██████    | 240748/400000 [00:30<00:19, 8139.42it/s] 60%|██████    | 241563/400000 [00:30<00:19, 8034.80it/s] 61%|██████    | 242368/400000 [00:30<00:19, 7938.72it/s] 61%|██████    | 243163/400000 [00:31<00:20, 7778.10it/s] 61%|██████    | 244000/400000 [00:31<00:19, 7942.88it/s] 61%|██████    | 244817/400000 [00:31<00:19, 8006.85it/s] 61%|██████▏   | 245620/400000 [00:31<00:19, 7851.65it/s] 62%|██████▏   | 246407/400000 [00:31<00:19, 7733.89it/s] 62%|██████▏   | 247212/400000 [00:31<00:19, 7826.08it/s] 62%|██████▏   | 248064/400000 [00:31<00:18, 8021.18it/s] 62%|██████▏   | 248869/400000 [00:31<00:18, 8003.61it/s] 62%|██████▏   | 249671/400000 [00:31<00:19, 7884.01it/s] 63%|██████▎   | 250461/400000 [00:31<00:19, 7804.73it/s] 63%|██████▎   | 251293/400000 [00:32<00:18, 7949.53it/s] 63%|██████▎   | 252117/400000 [00:32<00:18, 8034.36it/s] 63%|██████▎   | 252960/400000 [00:32<00:18, 8148.18it/s] 63%|██████▎   | 253794/400000 [00:32<00:17, 8202.56it/s] 64%|██████▎   | 254616/400000 [00:32<00:18, 8047.64it/s] 64%|██████▍   | 255449/400000 [00:32<00:17, 8130.09it/s] 64%|██████▍   | 256288/400000 [00:32<00:17, 8205.48it/s] 64%|██████▍   | 257110/400000 [00:32<00:17, 8170.61it/s] 64%|██████▍   | 257939/400000 [00:32<00:17, 8205.96it/s] 65%|██████▍   | 258761/400000 [00:32<00:17, 8014.21it/s] 65%|██████▍   | 259565/400000 [00:33<00:17, 8020.14it/s] 65%|██████▌   | 260435/400000 [00:33<00:16, 8211.76it/s] 65%|██████▌   | 261266/400000 [00:33<00:16, 8239.32it/s] 66%|██████▌   | 262101/400000 [00:33<00:16, 8269.88it/s] 66%|██████▌   | 262929/400000 [00:33<00:16, 8254.32it/s] 66%|██████▌   | 263756/400000 [00:33<00:16, 8170.04it/s] 66%|██████▌   | 264588/400000 [00:33<00:16, 8214.35it/s] 66%|██████▋   | 265410/400000 [00:33<00:16, 8214.04it/s] 67%|██████▋   | 266262/400000 [00:33<00:16, 8303.11it/s] 67%|██████▋   | 267093/400000 [00:33<00:16, 8182.11it/s] 67%|██████▋   | 267912/400000 [00:34<00:16, 8146.25it/s] 67%|██████▋   | 268728/400000 [00:34<00:16, 8030.50it/s] 67%|██████▋   | 269585/400000 [00:34<00:15, 8183.66it/s] 68%|██████▊   | 270451/400000 [00:34<00:15, 8319.10it/s] 68%|██████▊   | 271285/400000 [00:34<00:15, 8128.54it/s] 68%|██████▊   | 272109/400000 [00:34<00:15, 8160.19it/s] 68%|██████▊   | 272927/400000 [00:34<00:15, 8066.40it/s] 68%|██████▊   | 273789/400000 [00:34<00:15, 8222.72it/s] 69%|██████▊   | 274613/400000 [00:34<00:15, 8076.07it/s] 69%|██████▉   | 275423/400000 [00:34<00:15, 8012.73it/s] 69%|██████▉   | 276246/400000 [00:35<00:15, 8076.45it/s] 69%|██████▉   | 277079/400000 [00:35<00:15, 8150.60it/s] 69%|██████▉   | 277930/400000 [00:35<00:14, 8253.45it/s] 70%|██████▉   | 278772/400000 [00:35<00:14, 8300.80it/s] 70%|██████▉   | 279603/400000 [00:35<00:14, 8197.91it/s] 70%|███████   | 280471/400000 [00:35<00:14, 8335.81it/s] 70%|███████   | 281331/400000 [00:35<00:14, 8412.15it/s] 71%|███████   | 282211/400000 [00:35<00:13, 8522.77it/s] 71%|███████   | 283081/400000 [00:35<00:13, 8573.99it/s] 71%|███████   | 283940/400000 [00:36<00:13, 8404.13it/s] 71%|███████   | 284782/400000 [00:36<00:14, 8169.75it/s] 71%|███████▏  | 285602/400000 [00:36<00:14, 8066.12it/s] 72%|███████▏  | 286411/400000 [00:36<00:14, 8043.41it/s] 72%|███████▏  | 287228/400000 [00:36<00:13, 8077.97it/s] 72%|███████▏  | 288037/400000 [00:36<00:14, 7963.43it/s] 72%|███████▏  | 288869/400000 [00:36<00:13, 8067.10it/s] 72%|███████▏  | 289724/400000 [00:36<00:13, 8203.78it/s] 73%|███████▎  | 290564/400000 [00:36<00:13, 8259.10it/s] 73%|███████▎  | 291391/400000 [00:36<00:13, 8058.54it/s] 73%|███████▎  | 292199/400000 [00:37<00:13, 7929.30it/s] 73%|███████▎  | 292998/400000 [00:37<00:13, 7947.23it/s] 73%|███████▎  | 293826/400000 [00:37<00:13, 8042.06it/s] 74%|███████▎  | 294642/400000 [00:37<00:13, 8074.72it/s] 74%|███████▍  | 295466/400000 [00:37<00:12, 8122.88it/s] 74%|███████▍  | 296279/400000 [00:37<00:12, 8066.06it/s] 74%|███████▍  | 297087/400000 [00:37<00:12, 7926.06it/s] 74%|███████▍  | 297881/400000 [00:37<00:13, 7708.46it/s] 75%|███████▍  | 298712/400000 [00:37<00:12, 7877.04it/s] 75%|███████▍  | 299502/400000 [00:37<00:12, 7879.89it/s] 75%|███████▌  | 300320/400000 [00:38<00:12, 7965.78it/s] 75%|███████▌  | 301165/400000 [00:38<00:12, 8104.10it/s] 76%|███████▌  | 302021/400000 [00:38<00:11, 8233.84it/s] 76%|███████▌  | 302882/400000 [00:38<00:11, 8343.16it/s] 76%|███████▌  | 303718/400000 [00:38<00:11, 8287.66it/s] 76%|███████▌  | 304548/400000 [00:38<00:11, 8078.50it/s] 76%|███████▋  | 305358/400000 [00:38<00:11, 8083.46it/s] 77%|███████▋  | 306177/400000 [00:38<00:11, 8111.96it/s] 77%|███████▋  | 306990/400000 [00:38<00:11, 8104.86it/s] 77%|███████▋  | 307819/400000 [00:38<00:11, 8158.14it/s] 77%|███████▋  | 308636/400000 [00:39<00:11, 8041.82it/s] 77%|███████▋  | 309458/400000 [00:39<00:11, 8094.07it/s] 78%|███████▊  | 310269/400000 [00:39<00:11, 7985.56it/s] 78%|███████▊  | 311069/400000 [00:39<00:11, 7873.55it/s] 78%|███████▊  | 311904/400000 [00:39<00:11, 8008.40it/s] 78%|███████▊  | 312707/400000 [00:39<00:10, 8001.44it/s] 78%|███████▊  | 313543/400000 [00:39<00:10, 8104.29it/s] 79%|███████▊  | 314395/400000 [00:39<00:10, 8223.90it/s] 79%|███████▉  | 315245/400000 [00:39<00:10, 8303.39it/s] 79%|███████▉  | 316115/400000 [00:39<00:09, 8417.00it/s] 79%|███████▉  | 316958/400000 [00:40<00:09, 8338.98it/s] 79%|███████▉  | 317793/400000 [00:40<00:09, 8315.06it/s] 80%|███████▉  | 318626/400000 [00:40<00:09, 8316.91it/s] 80%|███████▉  | 319485/400000 [00:40<00:09, 8395.35it/s] 80%|████████  | 320326/400000 [00:40<00:09, 8250.07it/s] 80%|████████  | 321152/400000 [00:40<00:09, 7977.32it/s] 80%|████████  | 321953/400000 [00:40<00:10, 7697.53it/s] 81%|████████  | 322727/400000 [00:40<00:10, 7660.56it/s] 81%|████████  | 323531/400000 [00:40<00:09, 7768.22it/s] 81%|████████  | 324368/400000 [00:41<00:09, 7937.25it/s] 81%|████████▏ | 325173/400000 [00:41<00:09, 7970.32it/s] 82%|████████▏ | 326014/400000 [00:41<00:09, 8095.18it/s] 82%|████████▏ | 326852/400000 [00:41<00:08, 8178.20it/s] 82%|████████▏ | 327672/400000 [00:41<00:08, 8126.14it/s] 82%|████████▏ | 328493/400000 [00:41<00:08, 8149.16it/s] 82%|████████▏ | 329309/400000 [00:41<00:08, 8017.54it/s] 83%|████████▎ | 330124/400000 [00:41<00:08, 8054.77it/s] 83%|████████▎ | 330950/400000 [00:41<00:08, 8110.95it/s] 83%|████████▎ | 331762/400000 [00:41<00:08, 8095.77it/s] 83%|████████▎ | 332620/400000 [00:42<00:08, 8234.68it/s] 83%|████████▎ | 333445/400000 [00:42<00:08, 7894.55it/s] 84%|████████▎ | 334253/400000 [00:42<00:08, 7949.10it/s] 84%|████████▍ | 335051/400000 [00:42<00:08, 7843.39it/s] 84%|████████▍ | 335838/400000 [00:42<00:08, 7601.20it/s] 84%|████████▍ | 336602/400000 [00:42<00:08, 7544.38it/s] 84%|████████▍ | 337419/400000 [00:42<00:08, 7719.14it/s] 85%|████████▍ | 338194/400000 [00:42<00:08, 7702.49it/s] 85%|████████▍ | 338967/400000 [00:42<00:07, 7659.48it/s] 85%|████████▍ | 339768/400000 [00:42<00:07, 7761.33it/s] 85%|████████▌ | 340637/400000 [00:43<00:07, 8017.10it/s] 85%|████████▌ | 341454/400000 [00:43<00:07, 8060.54it/s] 86%|████████▌ | 342308/400000 [00:43<00:07, 8198.47it/s] 86%|████████▌ | 343139/400000 [00:43<00:06, 8231.33it/s] 86%|████████▌ | 343964/400000 [00:43<00:06, 8217.79it/s] 86%|████████▌ | 344787/400000 [00:43<00:06, 8219.28it/s] 86%|████████▋ | 345620/400000 [00:43<00:06, 8251.71it/s] 87%|████████▋ | 346446/400000 [00:43<00:06, 8244.62it/s] 87%|████████▋ | 347271/400000 [00:43<00:06, 8223.45it/s] 87%|████████▋ | 348094/400000 [00:43<00:06, 8105.03it/s] 87%|████████▋ | 348906/400000 [00:44<00:06, 7970.24it/s] 87%|████████▋ | 349704/400000 [00:44<00:06, 7917.91it/s] 88%|████████▊ | 350536/400000 [00:44<00:06, 8032.18it/s] 88%|████████▊ | 351378/400000 [00:44<00:05, 8144.39it/s] 88%|████████▊ | 352209/400000 [00:44<00:05, 8191.34it/s] 88%|████████▊ | 353082/400000 [00:44<00:05, 8343.82it/s] 88%|████████▊ | 353923/400000 [00:44<00:05, 8360.92it/s] 89%|████████▊ | 354760/400000 [00:44<00:05, 8349.77it/s] 89%|████████▉ | 355596/400000 [00:44<00:05, 8349.17it/s] 89%|████████▉ | 356433/400000 [00:44<00:05, 8354.03it/s] 89%|████████▉ | 357296/400000 [00:45<00:05, 8433.35it/s] 90%|████████▉ | 358140/400000 [00:45<00:05, 8148.16it/s] 90%|████████▉ | 358958/400000 [00:45<00:05, 7961.18it/s] 90%|████████▉ | 359802/400000 [00:45<00:04, 8098.00it/s] 90%|█████████ | 360631/400000 [00:45<00:04, 8151.86it/s] 90%|█████████ | 361448/400000 [00:45<00:04, 8045.07it/s] 91%|█████████ | 362255/400000 [00:45<00:04, 7893.69it/s] 91%|█████████ | 363047/400000 [00:45<00:04, 7890.64it/s] 91%|█████████ | 363850/400000 [00:45<00:04, 7929.72it/s] 91%|█████████ | 364676/400000 [00:46<00:04, 8024.05it/s] 91%|█████████▏| 365519/400000 [00:46<00:04, 8141.15it/s] 92%|█████████▏| 366335/400000 [00:46<00:04, 8065.07it/s] 92%|█████████▏| 367158/400000 [00:46<00:04, 8113.31it/s] 92%|█████████▏| 367972/400000 [00:46<00:03, 8118.30it/s] 92%|█████████▏| 368785/400000 [00:46<00:04, 7735.02it/s] 92%|█████████▏| 369563/400000 [00:46<00:03, 7693.25it/s] 93%|█████████▎| 370373/400000 [00:46<00:03, 7810.24it/s] 93%|█████████▎| 371176/400000 [00:46<00:03, 7873.78it/s] 93%|█████████▎| 372031/400000 [00:46<00:03, 8063.33it/s] 93%|█████████▎| 372896/400000 [00:47<00:03, 8227.68it/s] 93%|█████████▎| 373722/400000 [00:47<00:03, 8151.70it/s] 94%|█████████▎| 374540/400000 [00:47<00:03, 8061.94it/s] 94%|█████████▍| 375348/400000 [00:47<00:03, 7716.39it/s] 94%|█████████▍| 376134/400000 [00:47<00:03, 7758.86it/s] 94%|█████████▍| 376940/400000 [00:47<00:02, 7845.63it/s] 94%|█████████▍| 377727/400000 [00:47<00:02, 7799.97it/s] 95%|█████████▍| 378534/400000 [00:47<00:02, 7877.00it/s] 95%|█████████▍| 379357/400000 [00:47<00:02, 7977.11it/s] 95%|█████████▌| 380156/400000 [00:47<00:02, 7960.92it/s] 95%|█████████▌| 380991/400000 [00:48<00:02, 8072.21it/s] 95%|█████████▌| 381800/400000 [00:48<00:02, 7835.25it/s] 96%|█████████▌| 382586/400000 [00:48<00:02, 7805.58it/s] 96%|█████████▌| 383400/400000 [00:48<00:02, 7902.31it/s] 96%|█████████▌| 384251/400000 [00:48<00:01, 8074.52it/s] 96%|█████████▋| 385111/400000 [00:48<00:01, 8223.16it/s] 96%|█████████▋| 385936/400000 [00:48<00:01, 8133.86it/s] 97%|█████████▋| 386752/400000 [00:48<00:01, 8132.66it/s] 97%|█████████▋| 387567/400000 [00:48<00:01, 8077.56it/s] 97%|█████████▋| 388376/400000 [00:48<00:01, 8045.63it/s] 97%|█████████▋| 389206/400000 [00:49<00:01, 8118.53it/s] 98%|█████████▊| 390019/400000 [00:49<00:01, 7980.85it/s] 98%|█████████▊| 390819/400000 [00:49<00:01, 7967.22it/s] 98%|█████████▊| 391656/400000 [00:49<00:01, 8081.43it/s] 98%|█████████▊| 392469/400000 [00:49<00:00, 8094.03it/s] 98%|█████████▊| 393285/400000 [00:49<00:00, 8112.61it/s] 99%|█████████▊| 394097/400000 [00:49<00:00, 8098.19it/s] 99%|█████████▊| 394908/400000 [00:49<00:00, 8069.37it/s] 99%|█████████▉| 395722/400000 [00:49<00:00, 8087.70it/s] 99%|█████████▉| 396531/400000 [00:50<00:00, 8086.17it/s] 99%|█████████▉| 397340/400000 [00:50<00:00, 8070.62it/s]100%|█████████▉| 398173/400000 [00:50<00:00, 8145.81it/s]100%|█████████▉| 398988/400000 [00:50<00:00, 8104.58it/s]100%|█████████▉| 399814/400000 [00:50<00:00, 8149.05it/s]100%|█████████▉| 399999/400000 [00:50<00:00, 7931.81it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f1dfa30a4a8>, <torchtext.data.dataset.TabularDataset object at 0x7f1dfa30a358>, <torchtext.vocab.Vocab object at 0x7f1dfa30a438>), {}) 

  




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

Dl Completed...:   0%|          | 0/4 [00:00<?, ? file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.76 file/s]Dl Completed...:  25%|██▌       | 1/4 [00:00<00:00,  8.76 file/s]Dl Completed...:  50%|█████     | 2/4 [00:00<00:00,  8.76 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.40 file/s]Dl Completed...:  75%|███████▌  | 3/4 [00:00<00:00, 10.40 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  4.51 file/s]Dl Completed...: 100%|██████████| 4/4 [00:00<00:00,  5.42 file/s]2020-07-26 14:31:15.862816: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-07-26 14:31:15.867137: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-07-26 14:31:15.867926: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x55a7a4d12f70 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-07-26 14:31:15.867942: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
[1mDownloading and preparing dataset mnist/3.0.1 (download: 11.06 MiB, generated: 21.00 MiB, total: 32.06 MiB) to /home/runner/tensorflow_datasets/mnist/3.0.1...[0m

[1mDataset mnist downloaded and prepared to /home/runner/tensorflow_datasets/mnist/3.0.1. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/ ['cifar10', 'test', 'fashion-mnist_small.npy', 'mnist2', 'mnist_dataset_small.npy', 'train'] 

  


 #################### get_dataset_torch 

  get_dataset_torch mlmodels/preprocess/generic:get_dataset_torch {'dataloader': 'torchvision.datasets:MNIST', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}}, 'shuffle': True, 'download': True} 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

0it [00:00, ?it/s]  0%|          | 16384/9912422 [00:00<01:07, 146149.77it/s] 69%|██████▉   | 6873088/9912422 [00:00<00:14, 208594.44it/s]9920512it [00:00, 41903596.90it/s]                           
0it [00:00, ?it/s]32768it [00:00, 1022124.53it/s]
0it [00:00, ?it/s]  1%|          | 16384/1648877 [00:00<00:10, 155198.29it/s] 69%|██████▉   | 1138688/1648877 [00:00<00:02, 220323.56it/s]1654784it [00:00, 6872478.42it/s]                            
0it [00:00, ?it/s]8192it [00:00, 209933.03it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/raw/train-images-idx3-ubyte.gz
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

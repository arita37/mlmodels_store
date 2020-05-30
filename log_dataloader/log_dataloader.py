
  test_dataloader /home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json Namespace(config_file='/home/runner/work/mlmodels/mlmodels/mlmodels/config/test_config.json', config_mode='test', do='test_dataloader', folder=None, log_file=None, name='ml_store', save_folder='ztest/') 

  ml_test --do test_dataloader 





 ********************************************************************************************************************************************

 ******** TAG ::  {'github_repo_url': 'https://github.com/arita37/mlmodels/tree/b0e42615ccdfb742ee65e40d94496863dcd8699c', 'url_branch_file': 'https://github.com/arita37/mlmodels/blob/adata2/', 'repo': 'arita37/mlmodels', 'branch': 'adata2', 'sha': 'b0e42615ccdfb742ee65e40d94496863dcd8699c', 'workflow': 'test_dataloader'}

 ******** GITHUB_WOKFLOW : https://github.com/arita37/mlmodels/actions?query=workflow%3Atest_dataloader

 ******** GITHUB_REPO_BRANCH : https://github.com/arita37/mlmodels/tree/adata2/

 ******** GITHUB_REPO_URL : https://github.com/arita37/mlmodels/tree/b0e42615ccdfb742ee65e40d94496863dcd8699c

 ******** GITHUB_COMMIT_URL : https://github.com/arita37/mlmodels/commit/b0e42615ccdfb742ee65e40d94496863dcd8699c

 ******** Click here for Online DEBUGGER : https://gitpod.io/#https://github.com/arita37/mlmodels/tree/b0e42615ccdfb742ee65e40d94496863dcd8699c

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

  
###### load_callable_from_uri LOADED <function split_xy_from_dict at 0x7f5102c7c598> 

  
 ######### postional parameters :  ['out'] 

  
 ######### Execute : preprocessor_func <function split_xy_from_dict at 0x7f5102c7c598> 

  URL:  sklearn.model_selection:train_test_split {'test_size': 0.5} 

  
###### load_callable_from_uri LOADED <function train_test_split at 0x7f516e2430d0> 

  
 ######### postional parameters :  [] 

  
 ######### Execute : preprocessor_func <function train_test_split at 0x7f516e2430d0> 

  URL:  mlmodels.dataloader:pickle_dump {'path': 'mlmodels/ztest/ml_keras/namentity_crm_bilstm/data.pkl'} 

  
###### load_callable_from_uri LOADED <function pickle_dump at 0x7f5187f95ea0> 

  
 ######### postional parameters :  ['t'] 

  
 ######### Execute : preprocessor_func <function pickle_dump at 0x7f5187f95ea0> 
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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f511babd8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f511babd8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f511babd8c8> , (data_info, **args) 

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
0it [00:00, ?it/s]  0%|          | 0/9912422 [00:00<?, ?it/s] 26%|██▌       | 2539520/9912422 [00:00<00:00, 25386387.43it/s]9920512it [00:00, 32744960.27it/s]                             
0it [00:00, ?it/s]  0%|          | 0/28881 [00:00<?, ?it/s]32768it [00:00, 254704.31it/s]           
0it [00:00, ?it/s]  0%|          | 0/1648877 [00:00<?, ?it/s]1654784it [00:00, 8240048.66it/s]          
0it [00:00, ?it/s]8192it [00:00, 154159.04it/s]Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz to dataset/vision/MNIST/MNIST/raw/train-images-idx3-ubyte.gz
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

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f51191eacc0>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f51191e04e0>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function tf_dataset_download at 0x7f511babd510> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function tf_dataset_download at 0x7f511babd510> 

  function with postional parmater data_info <function tf_dataset_download at 0x7f511babd510> , (data_info, **args) 

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
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 1/162 [00:00<01:09,  2.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   1%|          | 2/162 [00:00<01:09,  2.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 3/162 [00:00<01:08,  2.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   2%|▏         | 4/162 [00:00<01:08,  2.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   3%|▎         | 5/162 [00:00<01:07,  2.31 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   4%|▎         | 6/162 [00:00<00:48,  3.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▎         | 6/162 [00:00<00:48,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   4%|▍         | 7/162 [00:00<00:47,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   5%|▍         | 8/162 [00:00<00:47,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 9/162 [00:00<00:47,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   6%|▌         | 10/162 [00:00<00:46,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 11/162 [00:00<00:46,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   7%|▋         | 12/162 [00:00<00:46,  3.24 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:   8%|▊         | 13/162 [00:00<00:32,  4.52 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   8%|▊         | 13/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▊         | 14/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:   9%|▉         | 15/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|▉         | 16/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  10%|█         | 17/162 [00:00<00:32,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  11%|█         | 18/162 [00:00<00:31,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 19/162 [00:00<00:31,  4.52 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  12%|█▏        | 20/162 [00:00<00:22,  6.26 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  12%|█▏        | 20/162 [00:00<00:22,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  13%|█▎        | 21/162 [00:00<00:22,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▎        | 22/162 [00:00<00:22,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  14%|█▍        | 23/162 [00:00<00:22,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▍        | 24/162 [00:00<00:22,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  15%|█▌        | 25/162 [00:00<00:21,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  16%|█▌        | 26/162 [00:00<00:21,  6.26 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.57 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 27/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  17%|█▋        | 28/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  18%|█▊        | 29/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▊        | 30/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  19%|█▉        | 31/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|█▉        | 32/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  20%|██        | 33/162 [00:00<00:15,  8.57 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[A
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:00<?, ? url/s]
Dl Size...:  21%|██        | 34/162 [00:00<00:11, 11.56 MiB/s][A

Extraction completed...: 0 file [00:00, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 35/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  22%|██▏       | 36/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 37/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  23%|██▎       | 38/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  24%|██▍       | 39/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▍       | 40/162 [00:01<00:10, 11.56 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 15.31 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  25%|██▌       | 41/162 [00:01<00:07, 15.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  26%|██▌       | 42/162 [00:01<00:07, 15.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 43/162 [00:01<00:07, 15.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  27%|██▋       | 44/162 [00:01<00:07, 15.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 45/162 [00:01<00:07, 15.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  28%|██▊       | 46/162 [00:01<00:07, 15.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  29%|██▉       | 47/162 [00:01<00:07, 15.31 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.70 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|██▉       | 48/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  30%|███       | 49/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███       | 50/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  31%|███▏      | 51/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  32%|███▏      | 52/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 53/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  33%|███▎      | 54/162 [00:01<00:05, 19.70 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.82 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  34%|███▍      | 55/162 [00:01<00:04, 24.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▍      | 56/162 [00:01<00:04, 24.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  35%|███▌      | 57/162 [00:01<00:04, 24.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▌      | 58/162 [00:01<00:04, 24.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  36%|███▋      | 59/162 [00:01<00:04, 24.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  37%|███▋      | 60/162 [00:01<00:04, 24.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 61/162 [00:01<00:04, 24.82 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.24 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  38%|███▊      | 62/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  39%|███▉      | 63/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|███▉      | 64/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  40%|████      | 65/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████      | 66/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  41%|████▏     | 67/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  42%|████▏     | 68/162 [00:01<00:03, 30.24 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 35.71 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 69/162 [00:01<00:02, 35.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  43%|████▎     | 70/162 [00:01<00:02, 35.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 71/162 [00:01<00:02, 35.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  44%|████▍     | 72/162 [00:01<00:02, 35.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  45%|████▌     | 73/162 [00:01<00:02, 35.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▌     | 74/162 [00:01<00:02, 35.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  46%|████▋     | 75/162 [00:01<00:02, 35.71 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.77 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  47%|████▋     | 76/162 [00:01<00:02, 40.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 77/162 [00:01<00:02, 40.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  48%|████▊     | 78/162 [00:01<00:02, 40.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 79/162 [00:01<00:02, 40.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  49%|████▉     | 80/162 [00:01<00:02, 40.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  50%|█████     | 81/162 [00:01<00:01, 40.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 82/162 [00:01<00:01, 40.77 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.23 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  51%|█████     | 83/162 [00:01<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 84/162 [00:01<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  52%|█████▏    | 85/162 [00:01<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  53%|█████▎    | 86/162 [00:01<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▎    | 87/162 [00:01<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  54%|█████▍    | 88/162 [00:01<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  55%|█████▍    | 89/162 [00:01<00:01, 45.23 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 48.67 MiB/s][ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 90/162 [00:01<00:01, 48.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  56%|█████▌    | 91/162 [00:01<00:01, 48.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 92/162 [00:01<00:01, 48.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  57%|█████▋    | 93/162 [00:01<00:01, 48.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  58%|█████▊    | 94/162 [00:01<00:01, 48.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▊    | 95/162 [00:01<00:01, 48.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:01<?, ? url/s]
Dl Size...:  59%|█████▉    | 96/162 [00:01<00:01, 48.67 MiB/s][A

Extraction completed...: 0 file [00:01, ? file/s][A[A
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 52.56 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|█████▉    | 97/162 [00:02<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  60%|██████    | 98/162 [00:02<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  61%|██████    | 99/162 [00:02<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 100/162 [00:02<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  62%|██████▏   | 101/162 [00:02<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  63%|██████▎   | 102/162 [00:02<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▎   | 103/162 [00:02<00:01, 52.56 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 55.01 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  64%|██████▍   | 104/162 [00:02<00:01, 55.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▍   | 105/162 [00:02<00:01, 55.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  65%|██████▌   | 106/162 [00:02<00:01, 55.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  66%|██████▌   | 107/162 [00:02<00:00, 55.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 108/162 [00:02<00:00, 55.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  67%|██████▋   | 109/162 [00:02<00:00, 55.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  68%|██████▊   | 110/162 [00:02<00:00, 55.01 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 56.79 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▊   | 111/162 [00:02<00:00, 56.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  69%|██████▉   | 112/162 [00:02<00:00, 56.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|██████▉   | 113/162 [00:02<00:00, 56.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  70%|███████   | 114/162 [00:02<00:00, 56.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  71%|███████   | 115/162 [00:02<00:00, 56.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 116/162 [00:02<00:00, 56.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  72%|███████▏  | 117/162 [00:02<00:00, 56.79 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.94 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 118/162 [00:02<00:00, 58.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  73%|███████▎  | 119/162 [00:02<00:00, 58.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  74%|███████▍  | 120/162 [00:02<00:00, 58.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▍  | 121/162 [00:02<00:00, 58.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  75%|███████▌  | 122/162 [00:02<00:00, 58.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  76%|███████▌  | 123/162 [00:02<00:00, 58.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 124/162 [00:02<00:00, 58.94 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.05 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  77%|███████▋  | 125/162 [00:02<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 126/162 [00:02<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  78%|███████▊  | 127/162 [00:02<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  79%|███████▉  | 128/162 [00:02<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|███████▉  | 129/162 [00:02<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  80%|████████  | 130/162 [00:02<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████  | 131/162 [00:02<00:00, 60.05 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.19 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  81%|████████▏ | 132/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  82%|████████▏ | 133/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 134/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  83%|████████▎ | 135/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  84%|████████▍ | 136/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▍ | 137/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  85%|████████▌ | 138/162 [00:02<00:00, 61.19 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 61.30 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▌ | 139/162 [00:02<00:00, 61.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  86%|████████▋ | 140/162 [00:02<00:00, 61.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  87%|████████▋ | 141/162 [00:02<00:00, 61.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 142/162 [00:02<00:00, 61.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  88%|████████▊ | 143/162 [00:02<00:00, 61.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  89%|████████▉ | 144/162 [00:02<00:00, 61.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|████████▉ | 145/162 [00:02<00:00, 61.30 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.07 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  90%|█████████ | 146/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████ | 147/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  91%|█████████▏| 148/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  92%|█████████▏| 149/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 150/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  93%|█████████▎| 151/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 152/162 [00:02<00:00, 62.07 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 62.21 MiB/s][ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  94%|█████████▍| 153/162 [00:02<00:00, 62.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  95%|█████████▌| 154/162 [00:02<00:00, 62.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▌| 155/162 [00:02<00:00, 62.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  96%|█████████▋| 156/162 [00:02<00:00, 62.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  97%|█████████▋| 157/162 [00:02<00:00, 62.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 158/162 [00:02<00:00, 62.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:02<?, ? url/s]
Dl Size...:  98%|█████████▊| 159/162 [00:02<00:00, 62.21 MiB/s][A

Extraction completed...: 0 file [00:02, ? file/s][A[A
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 62.65 MiB/s][ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 160/162 [00:03<00:00, 62.65 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...:  99%|█████████▉| 161/162 [00:03<00:00, 62.65 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...:   0%|          | 0/1 [00:03<?, ? url/s]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 62.65 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.05s/ url]Dl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.05s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 62.65 MiB/s][A

Extraction completed...: 0 file [00:03, ? file/s][A[ADl Completed...: 100%|██████████| 1/1 [00:03<00:00,  3.05s/ url]
Dl Size...: 100%|██████████| 162/162 [00:03<00:00, 62.65 MiB/s][A

Extraction completed...:   0%|          | 0/1 [00:03<?, ? file/s][A[A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ file][A[ADl Completed...: 100%|██████████| 1/1 [00:05<00:00,  3.05s/ url]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 62.65 MiB/s][A

Extraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ file][A[AExtraction completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ file]
Dl Size...: 100%|██████████| 162/162 [00:05<00:00, 30.46 MiB/s]
Dl Completed...: 100%|██████████| 1/1 [00:05<00:00,  5.32s/ url]
0 examples [00:00, ? examples/s]2020-05-30 15:41:58.789010: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
2020-05-30 15:41:58.832446: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2294685000 Hz
2020-05-30 15:41:58.832738: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5627c92f3250 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-05-30 15:41:58.832777: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
1 examples [00:00,  7.44 examples/s]80 examples [00:00, 10.59 examples/s]171 examples [00:00, 15.05 examples/s]263 examples [00:00, 21.35 examples/s]354 examples [00:00, 30.20 examples/s]446 examples [00:00, 42.54 examples/s]540 examples [00:00, 59.61 examples/s]631 examples [00:00, 82.81 examples/s]715 examples [00:00, 113.41 examples/s]804 examples [00:01, 153.52 examples/s]893 examples [00:01, 204.11 examples/s]985 examples [00:01, 266.04 examples/s]1073 examples [00:01, 335.50 examples/s]1162 examples [00:01, 412.47 examples/s]1253 examples [00:01, 492.87 examples/s]1343 examples [00:01, 570.11 examples/s]1435 examples [00:01, 640.85 examples/s]1525 examples [00:01, 663.85 examples/s]1613 examples [00:01, 714.95 examples/s]1702 examples [00:02, 759.52 examples/s]1790 examples [00:02, 791.94 examples/s]1878 examples [00:02, 815.84 examples/s]1966 examples [00:02, 832.37 examples/s]2054 examples [00:02, 837.41 examples/s]2142 examples [00:02, 847.60 examples/s]2236 examples [00:02, 871.18 examples/s]2330 examples [00:02, 889.39 examples/s]2421 examples [00:02, 883.26 examples/s]2511 examples [00:02, 874.34 examples/s]2600 examples [00:03, 873.87 examples/s]2691 examples [00:03, 882.91 examples/s]2780 examples [00:03, 879.81 examples/s]2869 examples [00:03, 860.54 examples/s]2956 examples [00:03, 850.99 examples/s]3043 examples [00:03, 855.40 examples/s]3130 examples [00:03, 857.63 examples/s]3220 examples [00:03, 869.89 examples/s]3314 examples [00:03, 888.69 examples/s]3404 examples [00:04, 891.11 examples/s]3498 examples [00:04, 903.72 examples/s]3593 examples [00:04, 915.98 examples/s]3687 examples [00:04, 921.78 examples/s]3782 examples [00:04, 929.49 examples/s]3876 examples [00:04, 918.52 examples/s]3968 examples [00:04, 917.46 examples/s]4067 examples [00:04, 933.79 examples/s]4161 examples [00:04, 934.98 examples/s]4255 examples [00:04, 902.07 examples/s]4346 examples [00:05, 900.42 examples/s]4437 examples [00:05, 858.16 examples/s]4524 examples [00:05, 846.31 examples/s]4616 examples [00:05, 867.02 examples/s]4705 examples [00:05, 870.37 examples/s]4797 examples [00:05, 883.15 examples/s]4892 examples [00:05, 899.97 examples/s]4987 examples [00:05, 911.72 examples/s]5083 examples [00:05, 923.56 examples/s]5176 examples [00:05, 923.63 examples/s]5269 examples [00:06, 924.28 examples/s]5366 examples [00:06, 936.20 examples/s]5460 examples [00:06, 916.95 examples/s]5554 examples [00:06, 919.98 examples/s]5647 examples [00:06, 922.45 examples/s]5744 examples [00:06, 934.40 examples/s]5841 examples [00:06, 944.41 examples/s]5941 examples [00:06, 958.73 examples/s]6037 examples [00:06, 888.32 examples/s]6127 examples [00:07, 882.22 examples/s]6223 examples [00:07, 902.55 examples/s]6314 examples [00:07, 898.62 examples/s]6405 examples [00:07, 900.50 examples/s]6505 examples [00:07, 925.77 examples/s]6601 examples [00:07, 930.71 examples/s]6699 examples [00:07, 942.48 examples/s]6797 examples [00:07, 953.40 examples/s]6893 examples [00:07, 940.63 examples/s]6988 examples [00:07, 933.02 examples/s]7082 examples [00:08, 911.54 examples/s]7177 examples [00:08, 922.40 examples/s]7275 examples [00:08, 936.26 examples/s]7369 examples [00:08, 934.89 examples/s]7465 examples [00:08, 940.67 examples/s]7560 examples [00:08, 900.49 examples/s]7651 examples [00:08, 876.17 examples/s]7740 examples [00:08, 873.48 examples/s]7834 examples [00:08, 891.49 examples/s]7929 examples [00:08, 906.52 examples/s]8024 examples [00:09, 917.35 examples/s]8116 examples [00:09, 912.07 examples/s]8213 examples [00:09, 928.20 examples/s]8311 examples [00:09, 940.60 examples/s]8406 examples [00:09, 939.49 examples/s]8501 examples [00:09, 917.65 examples/s]8593 examples [00:09, 891.60 examples/s]8691 examples [00:09, 915.29 examples/s]8787 examples [00:09, 927.34 examples/s]8881 examples [00:09, 925.30 examples/s]8974 examples [00:10, 914.73 examples/s]9066 examples [00:10, 907.30 examples/s]9158 examples [00:10, 909.06 examples/s]9251 examples [00:10, 914.05 examples/s]9346 examples [00:10, 922.05 examples/s]9442 examples [00:10, 932.69 examples/s]9541 examples [00:10, 947.04 examples/s]9636 examples [00:10, 947.84 examples/s]9731 examples [00:10, 940.67 examples/s]9826 examples [00:11, 885.06 examples/s]9916 examples [00:11, 871.61 examples/s]10004 examples [00:11, 826.45 examples/s]10088 examples [00:11, 815.48 examples/s]10181 examples [00:11, 844.46 examples/s]10269 examples [00:11, 854.67 examples/s]10359 examples [00:11, 867.15 examples/s]10447 examples [00:11, 849.65 examples/s]10537 examples [00:11, 861.96 examples/s]10630 examples [00:11, 880.64 examples/s]10726 examples [00:12, 901.06 examples/s]10824 examples [00:12, 920.84 examples/s]10917 examples [00:12, 921.19 examples/s]11010 examples [00:12, 923.19 examples/s]11103 examples [00:12, 878.04 examples/s]11192 examples [00:12, 870.99 examples/s]11282 examples [00:12, 879.19 examples/s]11371 examples [00:12, 875.77 examples/s]11459 examples [00:12, 861.98 examples/s]11551 examples [00:13, 878.41 examples/s]11640 examples [00:13, 874.77 examples/s]11732 examples [00:13, 885.87 examples/s]11828 examples [00:13, 905.32 examples/s]11924 examples [00:13, 918.22 examples/s]12020 examples [00:13, 929.30 examples/s]12114 examples [00:13, 887.05 examples/s]12204 examples [00:13, 888.54 examples/s]12300 examples [00:13, 908.02 examples/s]12393 examples [00:13, 914.22 examples/s]12492 examples [00:14, 934.30 examples/s]12586 examples [00:14, 882.10 examples/s]12681 examples [00:14, 900.00 examples/s]12772 examples [00:14, 900.33 examples/s]12864 examples [00:14, 903.98 examples/s]12955 examples [00:14, 895.76 examples/s]13045 examples [00:14, 895.90 examples/s]13136 examples [00:14, 898.87 examples/s]13227 examples [00:14, 900.05 examples/s]13319 examples [00:14, 902.73 examples/s]13410 examples [00:15, 888.19 examples/s]13499 examples [00:15, 885.67 examples/s]13588 examples [00:15, 865.49 examples/s]13675 examples [00:15, 860.56 examples/s]13764 examples [00:15, 868.53 examples/s]13852 examples [00:15, 871.52 examples/s]13942 examples [00:15, 878.31 examples/s]14036 examples [00:15, 895.83 examples/s]14131 examples [00:15, 908.66 examples/s]14228 examples [00:15, 925.48 examples/s]14322 examples [00:16, 929.43 examples/s]14416 examples [00:16, 922.27 examples/s]14509 examples [00:16, 920.74 examples/s]14603 examples [00:16, 925.84 examples/s]14696 examples [00:16, 880.55 examples/s]14785 examples [00:16, 869.14 examples/s]14879 examples [00:16, 886.74 examples/s]14974 examples [00:16, 902.91 examples/s]15065 examples [00:16, 904.01 examples/s]15156 examples [00:17, 877.88 examples/s]15245 examples [00:17, 877.21 examples/s]15333 examples [00:17, 864.44 examples/s]15425 examples [00:17, 878.67 examples/s]15519 examples [00:17, 895.32 examples/s]15618 examples [00:17, 919.51 examples/s]15711 examples [00:17, 912.28 examples/s]15805 examples [00:17, 919.19 examples/s]15908 examples [00:17, 949.32 examples/s]16005 examples [00:17, 955.33 examples/s]16102 examples [00:18, 957.01 examples/s]16198 examples [00:18, 955.53 examples/s]16294 examples [00:18, 952.33 examples/s]16390 examples [00:18, 946.64 examples/s]16485 examples [00:18, 933.53 examples/s]16579 examples [00:18, 935.35 examples/s]16673 examples [00:18, 909.09 examples/s]16765 examples [00:18, 897.00 examples/s]16864 examples [00:18, 922.39 examples/s]16957 examples [00:18, 888.44 examples/s]17053 examples [00:19, 906.25 examples/s]17145 examples [00:19, 896.83 examples/s]17236 examples [00:19, 898.77 examples/s]17330 examples [00:19, 908.24 examples/s]17422 examples [00:19, 910.07 examples/s]17514 examples [00:19, 912.00 examples/s]17606 examples [00:19, 898.49 examples/s]17698 examples [00:19, 903.84 examples/s]17791 examples [00:19, 908.85 examples/s]17884 examples [00:19, 913.44 examples/s]17981 examples [00:20, 927.70 examples/s]18074 examples [00:20, 887.62 examples/s]18169 examples [00:20, 902.70 examples/s]18260 examples [00:20, 891.44 examples/s]18350 examples [00:20, 888.82 examples/s]18440 examples [00:20, 891.08 examples/s]18534 examples [00:20, 903.82 examples/s]18631 examples [00:20, 922.43 examples/s]18726 examples [00:20, 930.00 examples/s]18820 examples [00:21, 901.99 examples/s]18916 examples [00:21, 918.50 examples/s]19009 examples [00:21, 918.20 examples/s]19102 examples [00:21, 920.90 examples/s]19201 examples [00:21, 938.55 examples/s]19296 examples [00:21, 907.65 examples/s]19391 examples [00:21, 919.17 examples/s]19486 examples [00:21, 926.27 examples/s]19581 examples [00:21, 931.12 examples/s]19675 examples [00:21, 928.29 examples/s]19768 examples [00:22, 913.36 examples/s]19860 examples [00:22, 903.71 examples/s]19951 examples [00:22, 886.46 examples/s]20040 examples [00:22, 849.39 examples/s]20133 examples [00:22, 871.30 examples/s]20226 examples [00:22, 886.26 examples/s]20318 examples [00:22, 892.41 examples/s]20410 examples [00:22, 898.65 examples/s]20502 examples [00:22, 903.77 examples/s]20593 examples [00:22, 899.24 examples/s]20685 examples [00:23, 904.93 examples/s]20776 examples [00:23, 899.22 examples/s]20870 examples [00:23, 909.90 examples/s]20965 examples [00:23, 920.24 examples/s]21060 examples [00:23, 926.22 examples/s]21153 examples [00:23, 921.02 examples/s]21246 examples [00:23, 899.95 examples/s]21338 examples [00:23, 900.32 examples/s]21429 examples [00:23, 902.80 examples/s]21520 examples [00:24, 895.27 examples/s]21610 examples [00:24, 875.13 examples/s]21700 examples [00:24, 882.27 examples/s]21790 examples [00:24, 887.33 examples/s]21884 examples [00:24, 899.86 examples/s]21975 examples [00:24, 902.38 examples/s]22066 examples [00:24, 890.25 examples/s]22156 examples [00:24, 883.09 examples/s]22247 examples [00:24, 889.03 examples/s]22339 examples [00:24, 897.71 examples/s]22429 examples [00:25, 890.03 examples/s]22520 examples [00:25, 893.07 examples/s]22610 examples [00:25, 888.15 examples/s]22699 examples [00:25, 852.40 examples/s]22785 examples [00:25, 840.01 examples/s]22879 examples [00:25, 867.06 examples/s]22977 examples [00:25, 896.94 examples/s]23068 examples [00:25, 894.90 examples/s]23158 examples [00:25, 892.31 examples/s]23250 examples [00:25, 900.20 examples/s]23342 examples [00:26, 903.28 examples/s]23433 examples [00:26, 886.69 examples/s]23522 examples [00:26, 873.91 examples/s]23610 examples [00:26, 844.63 examples/s]23696 examples [00:26, 849.06 examples/s]23782 examples [00:26, 841.39 examples/s]23867 examples [00:26, 829.49 examples/s]23955 examples [00:26, 841.88 examples/s]24047 examples [00:26, 861.81 examples/s]24134 examples [00:27, 852.36 examples/s]24220 examples [00:27, 841.79 examples/s]24312 examples [00:27, 863.03 examples/s]24404 examples [00:27, 878.32 examples/s]24497 examples [00:27, 893.13 examples/s]24587 examples [00:27, 894.76 examples/s]24677 examples [00:27, 887.16 examples/s]24768 examples [00:27, 892.15 examples/s]24858 examples [00:27, 889.90 examples/s]24948 examples [00:27, 891.05 examples/s]25040 examples [00:28, 897.47 examples/s]25132 examples [00:28, 902.72 examples/s]25224 examples [00:28, 905.80 examples/s]25317 examples [00:28, 911.64 examples/s]25409 examples [00:28, 909.41 examples/s]25500 examples [00:28, 907.49 examples/s]25594 examples [00:28, 914.27 examples/s]25686 examples [00:28, 885.69 examples/s]25775 examples [00:28, 861.82 examples/s]25869 examples [00:28, 883.03 examples/s]25960 examples [00:29, 889.24 examples/s]26050 examples [00:29, 865.69 examples/s]26137 examples [00:29, 835.86 examples/s]26222 examples [00:29, 812.34 examples/s]26315 examples [00:29, 844.15 examples/s]26405 examples [00:29, 858.73 examples/s]26503 examples [00:29, 890.62 examples/s]26598 examples [00:29, 907.60 examples/s]26694 examples [00:29, 920.10 examples/s]26793 examples [00:29, 938.86 examples/s]26889 examples [00:30, 943.71 examples/s]26988 examples [00:30, 956.92 examples/s]27087 examples [00:30, 965.82 examples/s]27184 examples [00:30, 932.35 examples/s]27278 examples [00:30, 903.53 examples/s]27374 examples [00:30, 918.35 examples/s]27470 examples [00:30, 929.00 examples/s]27566 examples [00:30, 937.47 examples/s]27660 examples [00:30, 933.00 examples/s]27754 examples [00:31, 923.93 examples/s]27847 examples [00:31, 918.76 examples/s]27943 examples [00:31, 928.02 examples/s]28036 examples [00:31, 883.31 examples/s]28125 examples [00:31, 879.68 examples/s]28214 examples [00:31, 880.64 examples/s]28311 examples [00:31, 904.03 examples/s]28413 examples [00:31, 934.54 examples/s]28507 examples [00:31, 906.13 examples/s]28599 examples [00:31, 879.47 examples/s]28688 examples [00:32, 865.55 examples/s]28775 examples [00:32, 834.75 examples/s]28865 examples [00:32, 853.11 examples/s]28951 examples [00:32, 844.20 examples/s]29040 examples [00:32, 857.38 examples/s]29127 examples [00:32, 782.58 examples/s]29209 examples [00:32, 792.20 examples/s]29307 examples [00:32, 837.28 examples/s]29393 examples [00:32, 837.38 examples/s]29480 examples [00:33, 845.63 examples/s]29576 examples [00:33, 875.15 examples/s]29671 examples [00:33, 895.14 examples/s]29762 examples [00:33, 896.50 examples/s]29854 examples [00:33, 889.24 examples/s]29949 examples [00:33, 906.41 examples/s]30040 examples [00:33, 865.06 examples/s]30128 examples [00:33, 859.93 examples/s]30215 examples [00:33, 861.21 examples/s]30306 examples [00:33, 873.62 examples/s]30397 examples [00:34, 881.69 examples/s]30492 examples [00:34, 901.09 examples/s]30583 examples [00:34, 903.25 examples/s]30674 examples [00:34, 903.62 examples/s]30765 examples [00:34, 873.36 examples/s]30853 examples [00:34, 868.90 examples/s]30945 examples [00:34, 883.62 examples/s]31037 examples [00:34, 892.58 examples/s]31130 examples [00:34, 901.03 examples/s]31222 examples [00:34, 905.08 examples/s]31316 examples [00:35, 914.76 examples/s]31411 examples [00:35, 922.76 examples/s]31504 examples [00:35, 924.03 examples/s]31597 examples [00:35, 918.30 examples/s]31689 examples [00:35, 876.58 examples/s]31784 examples [00:35, 895.86 examples/s]31875 examples [00:35, 899.79 examples/s]31966 examples [00:35, 901.39 examples/s]32062 examples [00:35, 916.52 examples/s]32154 examples [00:35, 909.51 examples/s]32249 examples [00:36, 918.63 examples/s]32344 examples [00:36, 925.40 examples/s]32438 examples [00:36, 928.36 examples/s]32532 examples [00:36, 928.85 examples/s]32625 examples [00:36, 886.68 examples/s]32717 examples [00:36, 893.46 examples/s]32815 examples [00:36, 915.53 examples/s]32911 examples [00:36, 925.84 examples/s]33008 examples [00:36, 935.36 examples/s]33102 examples [00:37, 910.30 examples/s]33194 examples [00:37, 862.10 examples/s]33285 examples [00:37, 875.46 examples/s]33376 examples [00:37, 885.28 examples/s]33469 examples [00:37, 896.89 examples/s]33563 examples [00:37, 907.97 examples/s]33655 examples [00:37, 907.71 examples/s]33750 examples [00:37, 919.11 examples/s]33845 examples [00:37, 926.66 examples/s]33938 examples [00:37, 926.72 examples/s]34031 examples [00:38, 917.17 examples/s]34123 examples [00:38, 917.84 examples/s]34217 examples [00:38, 923.14 examples/s]34311 examples [00:38, 927.79 examples/s]34405 examples [00:38, 930.05 examples/s]34499 examples [00:38, 930.32 examples/s]34598 examples [00:38, 945.33 examples/s]34693 examples [00:38, 871.58 examples/s]34782 examples [00:38, 864.27 examples/s]34871 examples [00:38, 870.07 examples/s]34962 examples [00:39, 880.06 examples/s]35051 examples [00:39, 866.45 examples/s]35142 examples [00:39, 877.90 examples/s]35231 examples [00:39, 878.13 examples/s]35323 examples [00:39, 888.50 examples/s]35413 examples [00:39, 858.16 examples/s]35509 examples [00:39, 884.49 examples/s]35606 examples [00:39, 905.72 examples/s]35702 examples [00:39, 921.33 examples/s]35795 examples [00:40, 922.54 examples/s]35892 examples [00:40, 936.24 examples/s]35989 examples [00:40, 944.72 examples/s]36084 examples [00:40, 928.47 examples/s]36178 examples [00:40, 922.93 examples/s]36271 examples [00:40, 896.47 examples/s]36370 examples [00:40, 920.96 examples/s]36470 examples [00:40, 941.98 examples/s]36567 examples [00:40, 949.61 examples/s]36667 examples [00:40, 964.08 examples/s]36764 examples [00:41, 953.73 examples/s]36860 examples [00:41, 948.13 examples/s]36955 examples [00:41, 946.39 examples/s]37050 examples [00:41, 927.29 examples/s]37143 examples [00:41, 926.57 examples/s]37236 examples [00:41, 920.15 examples/s]37330 examples [00:41, 924.65 examples/s]37424 examples [00:41, 928.26 examples/s]37517 examples [00:41, 896.88 examples/s]37611 examples [00:41, 907.02 examples/s]37702 examples [00:42, 882.27 examples/s]37791 examples [00:42, 851.37 examples/s]37877 examples [00:42, 852.77 examples/s]37967 examples [00:42, 864.67 examples/s]38068 examples [00:42, 903.67 examples/s]38165 examples [00:42, 921.88 examples/s]38261 examples [00:42, 932.24 examples/s]38361 examples [00:42, 950.48 examples/s]38461 examples [00:42, 963.69 examples/s]38558 examples [00:43, 955.08 examples/s]38654 examples [00:43, 929.34 examples/s]38748 examples [00:43, 923.66 examples/s]38844 examples [00:43, 931.82 examples/s]38939 examples [00:43, 935.18 examples/s]39038 examples [00:43, 949.77 examples/s]39134 examples [00:43, 952.17 examples/s]39232 examples [00:43, 959.28 examples/s]39329 examples [00:43, 941.16 examples/s]39424 examples [00:43, 930.89 examples/s]39523 examples [00:44, 946.79 examples/s]39618 examples [00:44, 937.10 examples/s]39714 examples [00:44, 940.68 examples/s]39809 examples [00:44, 928.55 examples/s]39903 examples [00:44, 930.00 examples/s]39997 examples [00:44, 911.50 examples/s]40089 examples [00:44, 856.61 examples/s]40178 examples [00:44, 865.79 examples/s]40273 examples [00:44, 887.77 examples/s]40363 examples [00:44, 879.33 examples/s]40455 examples [00:45, 890.65 examples/s]40553 examples [00:45, 910.99 examples/s]40645 examples [00:45, 904.09 examples/s]40741 examples [00:45, 919.95 examples/s]40834 examples [00:45, 907.53 examples/s]40925 examples [00:45, 888.42 examples/s]41021 examples [00:45, 906.19 examples/s]41120 examples [00:45, 929.24 examples/s]41218 examples [00:45, 942.13 examples/s]41321 examples [00:45, 965.79 examples/s]41418 examples [00:46, 962.51 examples/s]41515 examples [00:46, 928.28 examples/s]41609 examples [00:46, 931.23 examples/s]41704 examples [00:46, 935.15 examples/s]41798 examples [00:46, 929.92 examples/s]41892 examples [00:46, 931.43 examples/s]41986 examples [00:46, 928.56 examples/s]42084 examples [00:46, 942.84 examples/s]42179 examples [00:46, 932.63 examples/s]42273 examples [00:47, 925.54 examples/s]42366 examples [00:47, 868.81 examples/s]42454 examples [00:47, 863.49 examples/s]42541 examples [00:47, 847.34 examples/s]42632 examples [00:47, 863.60 examples/s]42730 examples [00:47, 894.02 examples/s]42825 examples [00:47, 908.53 examples/s]42925 examples [00:47, 932.21 examples/s]43019 examples [00:47, 933.76 examples/s]43113 examples [00:47, 906.49 examples/s]43210 examples [00:48, 921.93 examples/s]43304 examples [00:48, 926.07 examples/s]43399 examples [00:48, 929.80 examples/s]43493 examples [00:48, 930.47 examples/s]43590 examples [00:48, 939.09 examples/s]43685 examples [00:48, 939.61 examples/s]43780 examples [00:48, 929.40 examples/s]43874 examples [00:48, 878.76 examples/s]43963 examples [00:48, 838.05 examples/s]44048 examples [00:49, 835.34 examples/s]44137 examples [00:49, 849.00 examples/s]44224 examples [00:49, 854.21 examples/s]44315 examples [00:49, 867.80 examples/s]44404 examples [00:49, 874.30 examples/s]44495 examples [00:49, 884.15 examples/s]44585 examples [00:49, 888.16 examples/s]44674 examples [00:49, 866.34 examples/s]44765 examples [00:49, 878.28 examples/s]44861 examples [00:49, 900.49 examples/s]44957 examples [00:50, 914.80 examples/s]45051 examples [00:50, 921.32 examples/s]45144 examples [00:50, 918.33 examples/s]45238 examples [00:50, 924.08 examples/s]45332 examples [00:50, 927.53 examples/s]45425 examples [00:50, 909.34 examples/s]45517 examples [00:50, 898.90 examples/s]45608 examples [00:50, 888.67 examples/s]45698 examples [00:50, 890.86 examples/s]45789 examples [00:50, 895.14 examples/s]45879 examples [00:51, 841.76 examples/s]45964 examples [00:51, 813.79 examples/s]46056 examples [00:51, 842.21 examples/s]46141 examples [00:51, 831.79 examples/s]46239 examples [00:51, 868.92 examples/s]46335 examples [00:51, 892.37 examples/s]46431 examples [00:51, 911.21 examples/s]46523 examples [00:51, 913.43 examples/s]46616 examples [00:51, 916.01 examples/s]46710 examples [00:52, 920.25 examples/s]46803 examples [00:52, 923.02 examples/s]46896 examples [00:52, 887.44 examples/s]46986 examples [00:52, 855.07 examples/s]47076 examples [00:52, 865.89 examples/s]47169 examples [00:52, 882.58 examples/s]47262 examples [00:52, 894.92 examples/s]47352 examples [00:52, 893.76 examples/s]47442 examples [00:52, 892.98 examples/s]47534 examples [00:52, 899.53 examples/s]47625 examples [00:53, 902.63 examples/s]47716 examples [00:53, 894.13 examples/s]47809 examples [00:53, 902.11 examples/s]47900 examples [00:53, 893.01 examples/s]47993 examples [00:53, 901.96 examples/s]48084 examples [00:53, 903.40 examples/s]48180 examples [00:53, 918.33 examples/s]48277 examples [00:53, 930.70 examples/s]48371 examples [00:53, 925.27 examples/s]48464 examples [00:53, 907.29 examples/s]48555 examples [00:54, 868.86 examples/s]48643 examples [00:54, 858.60 examples/s]48730 examples [00:54, 856.59 examples/s]48821 examples [00:54, 869.67 examples/s]48913 examples [00:54, 884.06 examples/s]49002 examples [00:54, 884.89 examples/s]49098 examples [00:54, 904.32 examples/s]49189 examples [00:54, 880.10 examples/s]49278 examples [00:54, 876.40 examples/s]49370 examples [00:54, 887.89 examples/s]49459 examples [00:55, 888.18 examples/s]49548 examples [00:55, 887.51 examples/s]49637 examples [00:55, 861.84 examples/s]49729 examples [00:55, 877.40 examples/s]49821 examples [00:55, 889.65 examples/s]49911 examples [00:55, 854.15 examples/s]49997 examples [00:55, 852.29 examples/s]                                           0%|          | 0/50000 [00:00<?, ? examples/s] 12%|█▏        | 6037/50000 [00:00<00:00, 60369.63 examples/s] 34%|███▍      | 17242/50000 [00:00<00:00, 70063.65 examples/s] 57%|█████▋    | 28395/50000 [00:00<00:00, 78858.28 examples/s] 79%|███████▉  | 39675/50000 [00:00<00:00, 86682.85 examples/s]                                                               0 examples [00:00, ? examples/s]66 examples [00:00, 659.88 examples/s]154 examples [00:00, 711.63 examples/s]235 examples [00:00, 733.97 examples/s]320 examples [00:00, 764.25 examples/s]412 examples [00:00, 804.47 examples/s]503 examples [00:00, 831.26 examples/s]593 examples [00:00, 849.83 examples/s]685 examples [00:00, 867.48 examples/s]769 examples [00:00, 848.60 examples/s]854 examples [00:01, 846.18 examples/s]938 examples [00:01, 838.65 examples/s]1025 examples [00:01, 847.37 examples/s]1117 examples [00:01, 866.70 examples/s]1205 examples [00:01, 869.28 examples/s]1298 examples [00:01, 884.87 examples/s]1387 examples [00:01, 707.76 examples/s]1477 examples [00:01, 755.38 examples/s]1574 examples [00:01, 808.69 examples/s]1671 examples [00:02, 849.52 examples/s]1760 examples [00:02, 857.26 examples/s]1849 examples [00:02, 861.66 examples/s]1939 examples [00:02, 871.93 examples/s]2033 examples [00:02, 890.64 examples/s]2124 examples [00:02, 889.69 examples/s]2216 examples [00:02, 896.20 examples/s]2307 examples [00:02, 876.66 examples/s]2396 examples [00:02, 864.90 examples/s]2484 examples [00:02, 867.73 examples/s]2572 examples [00:03, 865.31 examples/s]2659 examples [00:03, 858.28 examples/s]2745 examples [00:03, 852.47 examples/s]2831 examples [00:03, 850.67 examples/s]2924 examples [00:03, 872.05 examples/s]3020 examples [00:03, 895.80 examples/s]3110 examples [00:03, 894.70 examples/s]3200 examples [00:03, 875.67 examples/s]3288 examples [00:03, 873.98 examples/s]3376 examples [00:03, 864.70 examples/s]3469 examples [00:04, 881.05 examples/s]3565 examples [00:04, 900.67 examples/s]3658 examples [00:04, 908.07 examples/s]3753 examples [00:04, 917.72 examples/s]3845 examples [00:04, 910.12 examples/s]3937 examples [00:04, 878.03 examples/s]4032 examples [00:04, 898.40 examples/s]4125 examples [00:04, 906.48 examples/s]4218 examples [00:04, 911.64 examples/s]4310 examples [00:04, 893.39 examples/s]4400 examples [00:05, 892.06 examples/s]4491 examples [00:05, 897.09 examples/s]4581 examples [00:05, 877.89 examples/s]4671 examples [00:05, 884.01 examples/s]4762 examples [00:05, 891.59 examples/s]4854 examples [00:05, 898.43 examples/s]4947 examples [00:05, 907.09 examples/s]5040 examples [00:05, 911.98 examples/s]5132 examples [00:05, 905.78 examples/s]5223 examples [00:05, 899.99 examples/s]5314 examples [00:06, 901.05 examples/s]5405 examples [00:06, 861.29 examples/s]5492 examples [00:06, 849.27 examples/s]5584 examples [00:06, 866.24 examples/s]5682 examples [00:06, 896.77 examples/s]5780 examples [00:06, 919.19 examples/s]5873 examples [00:06, 890.71 examples/s]5964 examples [00:06, 894.47 examples/s]6056 examples [00:06, 900.84 examples/s]6147 examples [00:07, 886.36 examples/s]6242 examples [00:07, 902.21 examples/s]6336 examples [00:07, 910.56 examples/s]6429 examples [00:07, 915.60 examples/s]6525 examples [00:07, 926.65 examples/s]6622 examples [00:07, 937.13 examples/s]6716 examples [00:07, 935.16 examples/s]6810 examples [00:07, 930.24 examples/s]6904 examples [00:07, 924.58 examples/s]6997 examples [00:07, 909.56 examples/s]7090 examples [00:08, 914.60 examples/s]7184 examples [00:08, 920.47 examples/s]7277 examples [00:08, 917.69 examples/s]7369 examples [00:08, 907.76 examples/s]7460 examples [00:08, 900.16 examples/s]7554 examples [00:08, 909.78 examples/s]7646 examples [00:08, 901.65 examples/s]7737 examples [00:08, 896.06 examples/s]7831 examples [00:08, 906.65 examples/s]7922 examples [00:08, 897.90 examples/s]8016 examples [00:09, 907.91 examples/s]8109 examples [00:09, 912.74 examples/s]8201 examples [00:09, 890.50 examples/s]8296 examples [00:09, 905.42 examples/s]8389 examples [00:09, 911.84 examples/s]8481 examples [00:09, 901.99 examples/s]8572 examples [00:09, 904.06 examples/s]8665 examples [00:09, 908.87 examples/s]8762 examples [00:09, 926.13 examples/s]8855 examples [00:09, 926.84 examples/s]8948 examples [00:10, 923.04 examples/s]9041 examples [00:10, 920.93 examples/s]9134 examples [00:10, 912.62 examples/s]9229 examples [00:10, 921.06 examples/s]9322 examples [00:10, 912.29 examples/s]9414 examples [00:10, 911.13 examples/s]9506 examples [00:10, 913.67 examples/s]9598 examples [00:10, 915.54 examples/s]9695 examples [00:10, 929.26 examples/s]9788 examples [00:11, 923.86 examples/s]9881 examples [00:11, 924.81 examples/s]9974 examples [00:11, 907.79 examples/s]                                          0%|          | 0/10000 [00:00<?, ? examples/s]                                                [1mDownloading and preparing dataset cifar10/3.0.2 (download: 162.17 MiB, generated: 132.40 MiB, total: 294.58 MiB) to /home/runner/tensorflow_datasets/cifar10/3.0.2...[0m



Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSC90JK/cifar10-train.tfrecord
Shuffling and writing examples to /home/runner/tensorflow_datasets/cifar10/3.0.2.incompleteSC90JK/cifar10-test.tfrecord
[1mDataset cifar10 downloaded and prepared to /home/runner/tensorflow_datasets/cifar10/3.0.2. Subsequent calls will reuse this data.[0m

  ############## Saving train dataset ############################### 

  ############## Saving test dataset ############################### 

  Saved /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/ ['train', 'test'] 

  URL:  mlmodels.preprocess.generic:get_dataset_torch {'dataloader': 'mlmodels.preprocess.generic:NumpyDataset', 'to_image': True, 'transform': {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}}, 'shuffle': True, 'download': True} 

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f511babd8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f511babd8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f511babd8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_generic', 'pass_data_pars': False, 'arg': {'fixed_size': 256}} 

  #### Loading dataloader URI 

  dataset :  <class 'mlmodels.preprocess.generic.NumpyDataset'> 
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/train/cifar10.npz
Dataset File path :  /home/runner/work/mlmodels/mlmodels/mlmodels/dataset/vision/cifar10/test/cifar10.npz

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f51674ac358>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f50a4f5c978>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function get_dataset_torch at 0x7f511babd8c8> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function get_dataset_torch at 0x7f511babd8c8> 

  function with postional parmater data_info <function get_dataset_torch at 0x7f511babd8c8> , (data_info, **args) 

  #### If transformer URI is Provided {'uri': 'mlmodels.preprocess.image:torch_transform_mnist', 'pass_data_pars': False, 'arg': {}} 

  #### Loading dataloader URI 

  dataset :  <class 'torchvision.datasets.mnist.MNIST'> 

  
 #####  get_Data DataLoader  

  ((<mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5102b70278>, <mlmodels.preprocess.generic.Custom_DataLoader object at 0x7f5102b70320>), {}) 

  




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

  
###### load_callable_from_uri LOADED <function split_train_valid at 0x7f509c036268> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function split_train_valid at 0x7f509c036268> 

  function with postional parmater data_info <function split_train_valid at 0x7f509c036268> , (data_info, **args) 
Spliting original file to train/valid set...

  URL:  mlmodels.model_tch.textcnn:create_tabular_dataset {'lang': 'en', 'pretrained_emb': 'glove.6B.300d'} 

  
###### load_callable_from_uri LOADED <function create_tabular_dataset at 0x7f509c036378> 

  
 ######### postional parameters :  ['data_info'] 

  
 ######### Execute : preprocessor_func <function create_tabular_dataset at 0x7f509c036378> 

  function with postional parmater data_info <function create_tabular_dataset at 0x7f509c036378> , (data_info, **args) 
.vector_cache/glove.6B.zip: 0.00B [00:00, ?B/s].vector_cache/glove.6B.zip:   0%|          | 8.19k/862M [00:05<154:38:02, 1.55kB/s].vector_cache/glove.6B.zip:   0%|          | 49.2k/862M [00:05<108:28:22, 2.21kB/s].vector_cache/glove.6B.zip:   0%|          | 180k/862M [00:05<75:59:33, 3.15kB/s]  .vector_cache/glove.6B.zip:   0%|          | 762k/862M [00:05<53:10:31, 4.50kB/s].vector_cache/glove.6B.zip:   0%|          | 3.07M/862M [00:05<37:07:37, 6.43kB/s].vector_cache/glove.6B.zip:   1%|          | 8.31M/862M [00:05<25:49:54, 9.18kB/s].vector_cache/glove.6B.zip:   1%|▏         | 12.2M/862M [00:06<18:00:03, 13.1kB/s].vector_cache/glove.6B.zip:   2%|▏         | 17.7M/862M [00:06<12:31:15, 18.7kB/s].vector_cache/glove.6B.zip:   2%|▏         | 20.9M/862M [00:06<8:44:02, 26.8kB/s] .vector_cache/glove.6B.zip:   3%|▎         | 26.6M/862M [00:06<6:04:26, 38.2kB/s].vector_cache/glove.6B.zip:   4%|▎         | 32.3M/862M [00:06<4:13:26, 54.6kB/s].vector_cache/glove.6B.zip:   4%|▍         | 35.3M/862M [00:06<2:56:55, 77.9kB/s].vector_cache/glove.6B.zip:   5%|▍         | 40.2M/862M [00:06<2:03:10, 111kB/s] .vector_cache/glove.6B.zip:   5%|▌         | 43.9M/862M [00:06<1:25:57, 159kB/s].vector_cache/glove.6B.zip:   6%|▌         | 49.0M/862M [00:06<59:52, 226kB/s]  .vector_cache/glove.6B.zip:   6%|▌         | 52.5M/862M [00:07<42:31, 317kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.6M/862M [00:09<31:31, 426kB/s].vector_cache/glove.6B.zip:   7%|▋         | 56.9M/862M [00:09<24:24, 550kB/s].vector_cache/glove.6B.zip:   7%|▋         | 57.8M/862M [00:09<17:34, 763kB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.5M/862M [00:09<12:24, 1.08MB/s].vector_cache/glove.6B.zip:   7%|▋         | 60.8M/862M [00:11<35:57, 372kB/s] .vector_cache/glove.6B.zip:   7%|▋         | 61.1M/862M [00:11<26:47, 498kB/s].vector_cache/glove.6B.zip:   7%|▋         | 62.4M/862M [00:11<19:07, 697kB/s].vector_cache/glove.6B.zip:   8%|▊         | 64.9M/862M [00:13<16:02, 828kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.1M/862M [00:13<14:18, 928kB/s].vector_cache/glove.6B.zip:   8%|▊         | 65.8M/862M [00:13<10:39, 1.25MB/s].vector_cache/glove.6B.zip:   8%|▊         | 68.0M/862M [00:13<07:36, 1.74MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.1M/862M [00:15<11:26, 1.16MB/s].vector_cache/glove.6B.zip:   8%|▊         | 69.5M/862M [00:15<09:52, 1.34MB/s].vector_cache/glove.6B.zip:   8%|▊         | 71.0M/862M [00:15<07:14, 1.82MB/s].vector_cache/glove.6B.zip:   8%|▊         | 73.2M/862M [00:17<08:01, 1.64MB/s].vector_cache/glove.6B.zip:   9%|▊         | 73.6M/862M [00:17<06:58, 1.89MB/s].vector_cache/glove.6B.zip:   9%|▊         | 75.2M/862M [00:17<05:09, 2.54MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.3M/862M [00:19<06:43, 1.94MB/s].vector_cache/glove.6B.zip:   9%|▉         | 77.7M/862M [00:19<06:02, 2.16MB/s].vector_cache/glove.6B.zip:   9%|▉         | 79.3M/862M [00:19<04:29, 2.90MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.4M/862M [00:21<06:15, 2.08MB/s].vector_cache/glove.6B.zip:   9%|▉         | 81.8M/862M [00:21<05:42, 2.28MB/s].vector_cache/glove.6B.zip:  10%|▉         | 83.4M/862M [00:21<04:18, 3.01MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.6M/862M [00:23<06:04, 2.13MB/s].vector_cache/glove.6B.zip:  10%|▉         | 85.8M/862M [00:23<06:53, 1.88MB/s].vector_cache/glove.6B.zip:  10%|█         | 86.5M/862M [00:23<05:28, 2.36MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:23<03:57, 3.25MB/s].vector_cache/glove.6B.zip:  10%|█         | 89.7M/862M [00:25<12:26:40, 17.2kB/s].vector_cache/glove.6B.zip:  10%|█         | 90.1M/862M [00:25<8:43:43, 24.6kB/s] .vector_cache/glove.6B.zip:  11%|█         | 91.6M/862M [00:25<6:06:11, 35.1kB/s].vector_cache/glove.6B.zip:  11%|█         | 93.8M/862M [00:27<4:18:36, 49.5kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.0M/862M [00:27<3:03:35, 69.7kB/s].vector_cache/glove.6B.zip:  11%|█         | 94.8M/862M [00:27<2:09:01, 99.1kB/s].vector_cache/glove.6B.zip:  11%|█▏        | 97.9M/862M [00:29<1:32:01, 138kB/s] .vector_cache/glove.6B.zip:  11%|█▏        | 98.3M/862M [00:29<1:05:42, 194kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 99.8M/862M [00:29<46:13, 275kB/s]  .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:31<35:12, 360kB/s] .vector_cache/glove.6B.zip:  12%|█▏        | 102M/862M [00:31<27:14, 465kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 103M/862M [00:31<19:35, 646kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 105M/862M [00:31<13:51, 911kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 106M/862M [00:33<15:42, 802kB/s].vector_cache/glove.6B.zip:  12%|█▏        | 107M/862M [00:33<12:16, 1.03MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 108M/862M [00:33<08:53, 1.41MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 110M/862M [00:35<09:09, 1.37MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 111M/862M [00:35<07:28, 1.68MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 112M/862M [00:35<05:29, 2.28MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:35<04:02, 3.09MB/s].vector_cache/glove.6B.zip:  13%|█▎        | 114M/862M [00:37<1:10:22, 177kB/s].vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:37<51:46, 241kB/s]  .vector_cache/glove.6B.zip:  13%|█▎        | 115M/862M [00:37<36:44, 339kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 117M/862M [00:37<25:53, 480kB/s].vector_cache/glove.6B.zip:  14%|█▎        | 118M/862M [00:39<22:30, 551kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 120M/862M [00:39<15:57, 775kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:40<13:53, 887kB/s].vector_cache/glove.6B.zip:  14%|█▍        | 123M/862M [00:41<11:00, 1.12MB/s].vector_cache/glove.6B.zip:  14%|█▍        | 125M/862M [00:41<07:56, 1.55MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:42<08:26, 1.45MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 127M/862M [00:43<08:24, 1.46MB/s].vector_cache/glove.6B.zip:  15%|█▍        | 128M/862M [00:43<06:30, 1.88MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:44<06:30, 1.87MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 131M/862M [00:45<05:47, 2.10MB/s].vector_cache/glove.6B.zip:  15%|█▌        | 133M/862M [00:45<04:21, 2.79MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:53, 2.06MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 135M/862M [00:46<05:20, 2.27MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 137M/862M [00:47<04:02, 2.99MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<05:40, 2.13MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 139M/862M [00:48<06:25, 1.88MB/s].vector_cache/glove.6B.zip:  16%|█▌        | 140M/862M [00:49<05:07, 2.35MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:49<03:41, 3.24MB/s].vector_cache/glove.6B.zip:  17%|█▋        | 143M/862M [00:50<1:29:34, 134kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 144M/862M [00:50<1:03:53, 187kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 145M/862M [00:51<44:55, 266kB/s]  .vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<34:09, 349kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 147M/862M [00:52<26:19, 452kB/s].vector_cache/glove.6B.zip:  17%|█▋        | 148M/862M [00:53<19:00, 626kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 151M/862M [00:54<15:18, 774kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:54<13:08, 901kB/s].vector_cache/glove.6B.zip:  18%|█▊        | 152M/862M [00:55<09:42, 1.22MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 154M/862M [00:55<06:59, 1.69MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 155M/862M [00:56<08:41, 1.35MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 156M/862M [00:56<07:16, 1.62MB/s].vector_cache/glove.6B.zip:  18%|█▊        | 157M/862M [00:57<05:21, 2.20MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<06:29, 1.81MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 160M/862M [00:58<05:43, 2.04MB/s].vector_cache/glove.6B.zip:  19%|█▊        | 162M/862M [00:58<04:14, 2.75MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<05:44, 2.03MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 164M/862M [01:00<06:23, 1.82MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 165M/862M [01:00<05:03, 2.30MB/s].vector_cache/glove.6B.zip:  19%|█▉        | 168M/862M [01:02<05:24, 2.14MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 168M/862M [01:02<04:45, 2.43MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 170M/862M [01:02<03:33, 3.24MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:02<02:38, 4.34MB/s].vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<48:14, 238kB/s] .vector_cache/glove.6B.zip:  20%|█▉        | 172M/862M [01:04<36:05, 319kB/s].vector_cache/glove.6B.zip:  20%|██        | 173M/862M [01:04<25:48, 445kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<19:50, 576kB/s].vector_cache/glove.6B.zip:  20%|██        | 176M/862M [01:06<15:05, 758kB/s].vector_cache/glove.6B.zip:  21%|██        | 178M/862M [01:06<10:49, 1.05MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<10:11, 1.11MB/s].vector_cache/glove.6B.zip:  21%|██        | 180M/862M [01:08<09:27, 1.20MB/s].vector_cache/glove.6B.zip:  21%|██        | 181M/862M [01:08<07:08, 1.59MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:08<05:06, 2.22MB/s].vector_cache/glove.6B.zip:  21%|██▏       | 184M/862M [01:10<15:56, 709kB/s] .vector_cache/glove.6B.zip:  21%|██▏       | 185M/862M [01:10<12:19, 916kB/s].vector_cache/glove.6B.zip:  22%|██▏       | 186M/862M [01:10<08:53, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 188M/862M [01:12<08:50, 1.27MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<08:29, 1.32MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 189M/862M [01:12<06:25, 1.75MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 192M/862M [01:12<04:38, 2.41MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<08:58, 1.24MB/s].vector_cache/glove.6B.zip:  22%|██▏       | 193M/862M [01:14<07:25, 1.50MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 194M/862M [01:14<05:28, 2.03MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<06:30, 1.70MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 197M/862M [01:16<05:14, 2.11MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 199M/862M [01:16<03:52, 2.85MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<06:23, 1.73MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 201M/862M [01:18<06:42, 1.64MB/s].vector_cache/glove.6B.zip:  23%|██▎       | 202M/862M [01:18<05:15, 2.10MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:20<05:26, 2.01MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 205M/862M [01:20<04:56, 2.21MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 207M/862M [01:20<03:44, 2.92MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:22<05:08, 2.12MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 209M/862M [01:22<04:31, 2.40MB/s].vector_cache/glove.6B.zip:  24%|██▍       | 211M/862M [01:22<03:23, 3.20MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:22<02:30, 4.30MB/s].vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:24<45:23, 238kB/s] .vector_cache/glove.6B.zip:  25%|██▍       | 213M/862M [01:24<33:59, 318kB/s].vector_cache/glove.6B.zip:  25%|██▍       | 214M/862M [01:24<24:13, 446kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 216M/862M [01:24<17:02, 632kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 217M/862M [01:26<18:05, 594kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 218M/862M [01:26<13:46, 780kB/s].vector_cache/glove.6B.zip:  25%|██▌       | 219M/862M [01:26<09:53, 1.08MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 221M/862M [01:28<09:24, 1.14MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:28<08:50, 1.21MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 222M/862M [01:28<06:40, 1.60MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 224M/862M [01:28<04:50, 2.20MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 225M/862M [01:30<06:56, 1.53MB/s].vector_cache/glove.6B.zip:  26%|██▌       | 226M/862M [01:30<05:57, 1.78MB/s].vector_cache/glove.6B.zip:  26%|██▋       | 227M/862M [01:30<04:23, 2.41MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:32<05:32, 1.90MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 230M/862M [01:32<06:02, 1.75MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 231M/862M [01:32<04:45, 2.21MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:34<05:00, 2.09MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 234M/862M [01:34<04:23, 2.38MB/s].vector_cache/glove.6B.zip:  27%|██▋       | 236M/862M [01:34<03:17, 3.18MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:34<02:26, 4.27MB/s].vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:36<43:39, 238kB/s] .vector_cache/glove.6B.zip:  28%|██▊       | 238M/862M [01:36<32:40, 318kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 239M/862M [01:36<23:17, 446kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 241M/862M [01:36<16:24, 631kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:37<15:42, 658kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 242M/862M [01:38<12:03, 856kB/s].vector_cache/glove.6B.zip:  28%|██▊       | 244M/862M [01:38<08:41, 1.19MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:39<08:28, 1.21MB/s].vector_cache/glove.6B.zip:  29%|██▊       | 246M/862M [01:40<06:58, 1.47MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 248M/862M [01:40<05:07, 2.00MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:41<05:59, 1.70MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 250M/862M [01:42<06:16, 1.63MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 251M/862M [01:42<04:54, 2.08MB/s].vector_cache/glove.6B.zip:  29%|██▉       | 254M/862M [01:43<05:03, 2.00MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 255M/862M [01:43<04:35, 2.21MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 256M/862M [01:44<03:25, 2.95MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 258M/862M [01:45<04:45, 2.11MB/s].vector_cache/glove.6B.zip:  30%|██▉       | 259M/862M [01:45<05:22, 1.87MB/s].vector_cache/glove.6B.zip:  30%|███       | 259M/862M [01:46<04:13, 2.38MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:46<03:03, 3.27MB/s].vector_cache/glove.6B.zip:  30%|███       | 262M/862M [01:47<12:38, 790kB/s] .vector_cache/glove.6B.zip:  30%|███       | 263M/862M [01:47<09:53, 1.01MB/s].vector_cache/glove.6B.zip:  31%|███       | 264M/862M [01:48<07:07, 1.40MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<07:18, 1.36MB/s].vector_cache/glove.6B.zip:  31%|███       | 267M/862M [01:49<07:07, 1.39MB/s].vector_cache/glove.6B.zip:  31%|███       | 268M/862M [01:49<05:24, 1.83MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 270M/862M [01:50<03:54, 2.53MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<08:13, 1.20MB/s].vector_cache/glove.6B.zip:  31%|███▏      | 271M/862M [01:51<06:45, 1.46MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 273M/862M [01:51<04:56, 1.99MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<05:45, 1.70MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 275M/862M [01:53<05:01, 1.94MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 277M/862M [01:53<03:45, 2.60MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<04:55, 1.97MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 279M/862M [01:55<05:25, 1.79MB/s].vector_cache/glove.6B.zip:  32%|███▏      | 280M/862M [01:55<04:17, 2.26MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<04:33, 2.12MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 283M/862M [01:57<04:11, 2.30MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 285M/862M [01:57<03:08, 3.07MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 287M/862M [01:59<04:26, 2.16MB/s].vector_cache/glove.6B.zip:  33%|███▎      | 288M/862M [01:59<03:56, 2.43MB/s].vector_cache/glove.6B.zip:  34%|███▎      | 289M/862M [01:59<02:59, 3.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<04:21, 2.19MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 291M/862M [02:01<04:59, 1.91MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 292M/862M [02:01<03:53, 2.44MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 294M/862M [02:01<02:50, 3.32MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 295M/862M [02:03<06:24, 1.47MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 296M/862M [02:03<05:27, 1.73MB/s].vector_cache/glove.6B.zip:  34%|███▍      | 297M/862M [02:03<04:03, 2.32MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 299M/862M [02:05<05:01, 1.86MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<05:25, 1.73MB/s].vector_cache/glove.6B.zip:  35%|███▍      | 300M/862M [02:05<04:12, 2.23MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 302M/862M [02:05<03:04, 3.03MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<05:56, 1.57MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 304M/862M [02:07<05:06, 1.82MB/s].vector_cache/glove.6B.zip:  35%|███▌      | 306M/862M [02:07<03:45, 2.46MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<04:48, 1.92MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 308M/862M [02:09<05:15, 1.76MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 309M/862M [02:09<04:08, 2.23MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<04:22, 2.10MB/s].vector_cache/glove.6B.zip:  36%|███▌      | 312M/862M [02:11<04:00, 2.28MB/s].vector_cache/glove.6B.zip:  36%|███▋      | 314M/862M [02:11<03:00, 3.05MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<04:12, 2.16MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 316M/862M [02:13<04:43, 1.93MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 317M/862M [02:13<03:41, 2.46MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 319M/862M [02:13<02:41, 3.36MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<07:17, 1.24MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 320M/862M [02:15<06:02, 1.49MB/s].vector_cache/glove.6B.zip:  37%|███▋      | 322M/862M [02:15<04:27, 2.02MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 324M/862M [02:17<05:12, 1.72MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 325M/862M [02:17<04:25, 2.03MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 326M/862M [02:17<03:15, 2.74MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:17<02:24, 3.70MB/s].vector_cache/glove.6B.zip:  38%|███▊      | 328M/862M [02:19<41:25, 215kB/s] .vector_cache/glove.6B.zip:  38%|███▊      | 330M/862M [02:19<29:04, 305kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 332M/862M [02:21<22:17, 396kB/s].vector_cache/glove.6B.zip:  39%|███▊      | 333M/862M [02:21<16:29, 535kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 334M/862M [02:21<11:44, 750kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<10:15, 854kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<08:57, 978kB/s].vector_cache/glove.6B.zip:  39%|███▉      | 337M/862M [02:23<06:39, 1.31MB/s].vector_cache/glove.6B.zip:  39%|███▉      | 340M/862M [02:23<04:45, 1.83MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:25<07:49, 1.11MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 341M/862M [02:25<06:21, 1.37MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 343M/862M [02:25<04:39, 1.86MB/s].vector_cache/glove.6B.zip:  40%|███▉      | 345M/862M [02:27<05:16, 1.63MB/s].vector_cache/glove.6B.zip:  40%|████      | 345M/862M [02:27<05:33, 1.55MB/s].vector_cache/glove.6B.zip:  40%|████      | 346M/862M [02:27<04:19, 1.99MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:27<03:06, 2.75MB/s].vector_cache/glove.6B.zip:  40%|████      | 349M/862M [02:28<8:12:09, 17.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 349M/862M [02:29<5:45:08, 24.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 351M/862M [02:29<4:01:04, 35.4kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:30<2:50:02, 49.9kB/s].vector_cache/glove.6B.zip:  41%|████      | 353M/862M [02:31<1:59:49, 70.8kB/s].vector_cache/glove.6B.zip:  41%|████      | 355M/862M [02:31<1:23:48, 101kB/s] .vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:32<1:00:23, 139kB/s].vector_cache/glove.6B.zip:  41%|████▏     | 357M/862M [02:33<43:57, 191kB/s]  .vector_cache/glove.6B.zip:  42%|████▏     | 358M/862M [02:33<31:09, 270kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 361M/862M [02:34<23:01, 363kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 362M/862M [02:34<16:58, 492kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 363M/862M [02:35<12:03, 690kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 365M/862M [02:36<10:20, 800kB/s].vector_cache/glove.6B.zip:  42%|████▏     | 366M/862M [02:36<07:56, 1.04MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 367M/862M [02:37<05:45, 1.43MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 369M/862M [02:38<05:57, 1.38MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:38<05:50, 1.40MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 370M/862M [02:39<04:30, 1.82MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<04:27, 1.83MB/s].vector_cache/glove.6B.zip:  43%|████▎     | 374M/862M [02:40<03:56, 2.06MB/s].vector_cache/glove.6B.zip:  44%|████▎     | 375M/862M [02:41<02:57, 2.74MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<03:57, 2.04MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 378M/862M [02:42<03:35, 2.25MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 380M/862M [02:42<02:40, 3.00MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<03:46, 2.13MB/s].vector_cache/glove.6B.zip:  44%|████▍     | 382M/862M [02:44<03:27, 2.32MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 384M/862M [02:44<02:36, 3.05MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<03:42, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 386M/862M [02:46<03:24, 2.33MB/s].vector_cache/glove.6B.zip:  45%|████▍     | 388M/862M [02:46<02:33, 3.09MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<03:39, 2.15MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 390M/862M [02:48<03:21, 2.34MB/s].vector_cache/glove.6B.zip:  45%|████▌     | 392M/862M [02:48<02:32, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 394M/862M [02:50<03:37, 2.16MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 395M/862M [02:50<03:19, 2.34MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 396M/862M [02:50<02:31, 3.08MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<03:35, 2.15MB/s].vector_cache/glove.6B.zip:  46%|████▌     | 398M/862M [02:52<04:05, 1.89MB/s].vector_cache/glove.6B.zip:  46%|████▋     | 399M/862M [02:52<03:11, 2.41MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:52<02:19, 3.31MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 402M/862M [02:54<06:53, 1.11MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 403M/862M [02:54<05:35, 1.37MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 404M/862M [02:54<04:06, 1.86MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 406M/862M [02:56<04:38, 1.64MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<04:51, 1.56MB/s].vector_cache/glove.6B.zip:  47%|████▋     | 407M/862M [02:56<03:47, 2.00MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:56<02:43, 2.76MB/s].vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:58<49:54, 151kB/s] .vector_cache/glove.6B.zip:  48%|████▊     | 411M/862M [02:58<35:41, 211kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 413M/862M [02:58<25:05, 299kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:00<19:13, 388kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 415M/862M [03:00<14:04, 529kB/s].vector_cache/glove.6B.zip:  48%|████▊     | 416M/862M [03:00<09:59, 744kB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:00<07:03, 1.05MB/s].vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:02<32:55, 224kB/s] .vector_cache/glove.6B.zip:  49%|████▊     | 419M/862M [03:02<23:46, 311kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 421M/862M [03:02<16:44, 440kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:04<13:24, 546kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 423M/862M [03:04<10:52, 673kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 424M/862M [03:04<07:54, 924kB/s].vector_cache/glove.6B.zip:  49%|████▉     | 426M/862M [03:04<05:37, 1.29MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:06<06:53, 1.05MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 427M/862M [03:06<05:34, 1.30MB/s].vector_cache/glove.6B.zip:  50%|████▉     | 429M/862M [03:06<04:04, 1.77MB/s].vector_cache/glove.6B.zip:  50%|█████     | 431M/862M [03:08<04:31, 1.59MB/s].vector_cache/glove.6B.zip:  50%|█████     | 432M/862M [03:08<03:46, 1.90MB/s].vector_cache/glove.6B.zip:  50%|█████     | 433M/862M [03:08<02:47, 2.57MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:08<02:02, 3.49MB/s].vector_cache/glove.6B.zip:  50%|█████     | 435M/862M [03:10<37:16, 191kB/s] .vector_cache/glove.6B.zip:  51%|█████     | 435M/862M [03:10<27:31, 258kB/s].vector_cache/glove.6B.zip:  51%|█████     | 436M/862M [03:10<19:35, 362kB/s].vector_cache/glove.6B.zip:  51%|█████     | 439M/862M [03:12<14:45, 477kB/s].vector_cache/glove.6B.zip:  51%|█████     | 440M/862M [03:12<11:02, 638kB/s].vector_cache/glove.6B.zip:  51%|█████     | 441M/862M [03:12<07:50, 894kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 443M/862M [03:14<07:06, 982kB/s].vector_cache/glove.6B.zip:  51%|█████▏    | 444M/862M [03:14<05:40, 1.23MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 445M/862M [03:14<04:06, 1.69MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:16<04:30, 1.53MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 448M/862M [03:16<03:51, 1.79MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 450M/862M [03:16<02:50, 2.42MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:18<03:36, 1.89MB/s].vector_cache/glove.6B.zip:  52%|█████▏    | 452M/862M [03:18<03:13, 2.12MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 454M/862M [03:18<02:23, 2.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:19<03:15, 2.08MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 456M/862M [03:20<03:39, 1.85MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 457M/862M [03:20<02:50, 2.38MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 459M/862M [03:20<02:04, 3.25MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:21<05:01, 1.33MB/s].vector_cache/glove.6B.zip:  53%|█████▎    | 460M/862M [03:22<04:12, 1.59MB/s].vector_cache/glove.6B.zip:  54%|█████▎    | 462M/862M [03:22<03:04, 2.17MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:23<03:42, 1.79MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 464M/862M [03:24<03:16, 2.03MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 466M/862M [03:24<02:25, 2.73MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 468M/862M [03:25<03:15, 2.02MB/s].vector_cache/glove.6B.zip:  54%|█████▍    | 469M/862M [03:25<02:56, 2.23MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 470M/862M [03:26<02:13, 2.95MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:27<03:04, 2.11MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 472M/862M [03:27<03:28, 1.87MB/s].vector_cache/glove.6B.zip:  55%|█████▍    | 473M/862M [03:28<02:43, 2.37MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 476M/862M [03:29<02:56, 2.18MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 477M/862M [03:29<02:36, 2.46MB/s].vector_cache/glove.6B.zip:  55%|█████▌    | 478M/862M [03:30<01:59, 3.22MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<02:53, 2.20MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 481M/862M [03:31<03:18, 1.92MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 482M/862M [03:31<02:35, 2.45MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 484M/862M [03:32<01:53, 3.33MB/s].vector_cache/glove.6B.zip:  56%|█████▌    | 485M/862M [03:33<04:05, 1.54MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 485M/862M [03:33<03:30, 1.79MB/s].vector_cache/glove.6B.zip:  56%|█████▋    | 487M/862M [03:33<02:36, 2.40MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:35<03:16, 1.90MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 489M/862M [03:35<02:55, 2.13MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 491M/862M [03:35<02:11, 2.82MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:37<02:59, 2.06MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 493M/862M [03:37<02:43, 2.26MB/s].vector_cache/glove.6B.zip:  57%|█████▋    | 495M/862M [03:37<02:03, 2.99MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:39<02:52, 2.12MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 497M/862M [03:39<03:12, 1.90MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 498M/862M [03:39<02:29, 2.43MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 500M/862M [03:39<01:50, 3.27MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:41<03:15, 1.85MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 501M/862M [03:41<02:54, 2.07MB/s].vector_cache/glove.6B.zip:  58%|█████▊    | 503M/862M [03:41<02:09, 2.78MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 505M/862M [03:43<02:53, 2.05MB/s].vector_cache/glove.6B.zip:  59%|█████▊    | 506M/862M [03:43<02:37, 2.26MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 507M/862M [03:43<01:58, 3.01MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 509M/862M [03:45<02:46, 2.12MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 510M/862M [03:45<02:32, 2.32MB/s].vector_cache/glove.6B.zip:  59%|█████▉    | 511M/862M [03:45<01:54, 3.05MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 513M/862M [03:47<02:42, 2.14MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<03:05, 1.88MB/s].vector_cache/glove.6B.zip:  60%|█████▉    | 514M/862M [03:47<02:27, 2.36MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:49<02:38, 2.18MB/s].vector_cache/glove.6B.zip:  60%|██████    | 518M/862M [03:49<02:26, 2.35MB/s].vector_cache/glove.6B.zip:  60%|██████    | 520M/862M [03:49<01:51, 3.09MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:51<02:37, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████    | 522M/862M [03:51<03:00, 1.88MB/s].vector_cache/glove.6B.zip:  61%|██████    | 523M/862M [03:51<02:23, 2.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:53<02:34, 2.18MB/s].vector_cache/glove.6B.zip:  61%|██████    | 526M/862M [03:53<02:22, 2.37MB/s].vector_cache/glove.6B.zip:  61%|██████    | 528M/862M [03:53<01:47, 3.11MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:55<02:32, 2.17MB/s].vector_cache/glove.6B.zip:  61%|██████▏   | 530M/862M [03:55<02:54, 1.90MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 531M/862M [03:55<02:19, 2.38MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:55<01:40, 3.26MB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:57<45:50, 119kB/s] .vector_cache/glove.6B.zip:  62%|██████▏   | 534M/862M [03:57<32:36, 168kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 536M/862M [03:57<22:50, 238kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:59<17:08, 315kB/s].vector_cache/glove.6B.zip:  62%|██████▏   | 538M/862M [03:59<13:05, 412kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 539M/862M [03:59<09:25, 571kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 542M/862M [04:01<07:24, 721kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 543M/862M [04:01<05:43, 930kB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 544M/862M [04:01<04:07, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 546M/862M [04:03<04:06, 1.28MB/s].vector_cache/glove.6B.zip:  63%|██████▎   | 547M/862M [04:03<03:24, 1.54MB/s].vector_cache/glove.6B.zip:  64%|██████▎   | 548M/862M [04:03<02:30, 2.09MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 550M/862M [04:05<02:58, 1.75MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:05<03:05, 1.68MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 551M/862M [04:05<02:23, 2.17MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 554M/862M [04:05<01:44, 2.96MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:07<03:30, 1.46MB/s].vector_cache/glove.6B.zip:  64%|██████▍   | 555M/862M [04:07<02:59, 1.71MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 557M/862M [04:07<02:12, 2.30MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:09<02:43, 1.86MB/s].vector_cache/glove.6B.zip:  65%|██████▍   | 559M/862M [04:09<02:25, 2.09MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 561M/862M [04:09<01:48, 2.77MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:10<02:26, 2.04MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 563M/862M [04:11<02:43, 1.83MB/s].vector_cache/glove.6B.zip:  65%|██████▌   | 564M/862M [04:11<02:06, 2.35MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 566M/862M [04:11<01:32, 3.22MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:12<03:55, 1.25MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 567M/862M [04:13<03:14, 1.52MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 569M/862M [04:13<02:22, 2.06MB/s].vector_cache/glove.6B.zip:  66%|██████▌   | 571M/862M [04:14<02:47, 1.74MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 571M/862M [04:15<02:56, 1.65MB/s].vector_cache/glove.6B.zip:  66%|██████▋   | 572M/862M [04:15<02:18, 2.10MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 575M/862M [04:16<02:22, 2.02MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 576M/862M [04:16<02:09, 2.22MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 577M/862M [04:17<01:36, 2.96MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 579M/862M [04:18<02:12, 2.13MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 580M/862M [04:18<02:01, 2.32MB/s].vector_cache/glove.6B.zip:  67%|██████▋   | 581M/862M [04:19<01:31, 3.06MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 583M/862M [04:20<02:09, 2.15MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:20<02:28, 1.88MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 584M/862M [04:21<01:57, 2.37MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 587M/862M [04:21<01:24, 3.25MB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<4:24:30, 17.3kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 588M/862M [04:22<3:05:22, 24.7kB/s].vector_cache/glove.6B.zip:  68%|██████▊   | 589M/862M [04:22<2:09:07, 35.2kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<1:30:43, 49.7kB/s].vector_cache/glove.6B.zip:  69%|██████▊   | 592M/862M [04:24<1:03:48, 70.6kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 593M/862M [04:24<44:37, 101kB/s]   .vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:25<30:59, 143kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<38:29, 115kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 596M/862M [04:26<27:49, 159kB/s].vector_cache/glove.6B.zip:  69%|██████▉   | 597M/862M [04:26<19:38, 225kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<14:18, 306kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 600M/862M [04:28<10:26, 418kB/s].vector_cache/glove.6B.zip:  70%|██████▉   | 602M/862M [04:28<07:22, 588kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<06:07, 702kB/s].vector_cache/glove.6B.zip:  70%|███████   | 604M/862M [04:30<04:43, 908kB/s].vector_cache/glove.6B.zip:  70%|███████   | 606M/862M [04:30<03:23, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:32<03:21, 1.26MB/s].vector_cache/glove.6B.zip:  71%|███████   | 608M/862M [04:32<03:13, 1.31MB/s].vector_cache/glove.6B.zip:  71%|███████   | 609M/862M [04:32<02:26, 1.73MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:32<01:44, 2.40MB/s].vector_cache/glove.6B.zip:  71%|███████   | 612M/862M [04:34<12:53, 323kB/s] .vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:34<09:26, 441kB/s].vector_cache/glove.6B.zip:  71%|███████   | 613M/862M [04:34<06:57, 596kB/s].vector_cache/glove.6B.zip:  71%|███████   | 614M/862M [04:35<05:11, 796kB/s].vector_cache/glove.6B.zip:  71%|███████▏  | 616M/862M [04:36<04:24, 931kB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 617M/862M [04:36<03:29, 1.17MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 618M/862M [04:36<02:32, 1.60MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 620M/862M [04:38<02:42, 1.49MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 621M/862M [04:38<02:18, 1.74MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 622M/862M [04:38<01:42, 2.34MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:40<02:07, 1.87MB/s].vector_cache/glove.6B.zip:  72%|███████▏  | 625M/862M [04:40<02:17, 1.73MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 626M/862M [04:40<01:47, 2.19MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:42<01:52, 2.08MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 629M/862M [04:42<01:38, 2.37MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 631M/862M [04:42<01:14, 3.11MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<01:46, 2.16MB/s].vector_cache/glove.6B.zip:  73%|███████▎  | 633M/862M [04:44<01:58, 1.93MB/s].vector_cache/glove.6B.zip:  74%|███████▎  | 634M/862M [04:44<01:34, 2.42MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:44<01:07, 3.32MB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:46<3:37:20, 17.3kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 637M/862M [04:46<2:32:12, 24.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 638M/862M [04:46<1:46:04, 35.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:46<1:13:32, 50.2kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<58:23, 63.1kB/s]  .vector_cache/glove.6B.zip:  74%|███████▍  | 641M/862M [04:48<41:35, 88.6kB/s].vector_cache/glove.6B.zip:  74%|███████▍  | 642M/862M [04:48<29:10, 126kB/s] .vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:48<20:11, 179kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<20:23, 177kB/s].vector_cache/glove.6B.zip:  75%|███████▍  | 645M/862M [04:50<14:37, 247kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 647M/862M [04:50<10:15, 350kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 649M/862M [04:52<07:56, 447kB/s].vector_cache/glove.6B.zip:  75%|███████▌  | 650M/862M [04:52<05:54, 599kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 651M/862M [04:52<04:11, 838kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 653M/862M [04:54<03:44, 932kB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:54<03:17, 1.06MB/s].vector_cache/glove.6B.zip:  76%|███████▌  | 654M/862M [04:54<02:28, 1.40MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 657M/862M [04:56<02:15, 1.51MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 658M/862M [04:56<01:56, 1.76MB/s].vector_cache/glove.6B.zip:  76%|███████▋  | 659M/862M [04:56<01:25, 2.38MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<01:46, 1.89MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<01:57, 1.71MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 662M/862M [04:58<01:29, 2.22MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 664M/862M [04:58<01:08, 2.91MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<01:32, 2.13MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 666M/862M [05:00<01:24, 2.32MB/s].vector_cache/glove.6B.zip:  77%|███████▋  | 668M/862M [05:00<01:03, 3.06MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:01<01:29, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 670M/862M [05:02<01:22, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 672M/862M [05:02<01:01, 3.08MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:03<01:27, 2.15MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 674M/862M [05:04<01:20, 2.34MB/s].vector_cache/glove.6B.zip:  78%|███████▊  | 676M/862M [05:04<01:00, 3.08MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<01:25, 2.16MB/s].vector_cache/glove.6B.zip:  79%|███████▊  | 678M/862M [05:05<01:15, 2.44MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 680M/862M [05:06<00:56, 3.22MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:06<00:41, 4.34MB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 682M/862M [05:07<12:34, 239kB/s] .vector_cache/glove.6B.zip:  79%|███████▉  | 683M/862M [05:07<09:05, 329kB/s].vector_cache/glove.6B.zip:  79%|███████▉  | 684M/862M [05:08<06:22, 465kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<05:06, 574kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 686M/862M [05:09<04:10, 703kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 687M/862M [05:10<03:02, 960kB/s].vector_cache/glove.6B.zip:  80%|███████▉  | 689M/862M [05:10<02:08, 1.34MB/s].vector_cache/glove.6B.zip:  80%|████████  | 690M/862M [05:11<02:49, 1.01MB/s].vector_cache/glove.6B.zip:  80%|████████  | 691M/862M [05:11<02:16, 1.26MB/s].vector_cache/glove.6B.zip:  80%|████████  | 692M/862M [05:12<01:38, 1.72MB/s].vector_cache/glove.6B.zip:  81%|████████  | 694M/862M [05:13<01:47, 1.56MB/s].vector_cache/glove.6B.zip:  81%|████████  | 695M/862M [05:13<01:32, 1.81MB/s].vector_cache/glove.6B.zip:  81%|████████  | 696M/862M [05:14<01:08, 2.43MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:25, 1.91MB/s].vector_cache/glove.6B.zip:  81%|████████  | 699M/862M [05:15<01:16, 2.13MB/s].vector_cache/glove.6B.zip:  81%|████████▏ | 701M/862M [05:15<00:57, 2.83MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:17, 2.06MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 703M/862M [05:17<01:10, 2.26MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 705M/862M [05:17<00:52, 2.98MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:13, 2.12MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 707M/862M [05:19<01:22, 1.87MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 708M/862M [05:19<01:05, 2.35MB/s].vector_cache/glove.6B.zip:  82%|████████▏ | 711M/862M [05:21<01:09, 2.18MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 711M/862M [05:21<01:01, 2.46MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 713M/862M [05:21<00:45, 3.28MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:21<00:33, 4.39MB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<10:15, 239kB/s] .vector_cache/glove.6B.zip:  83%|████████▎ | 715M/862M [05:23<07:39, 320kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 716M/862M [05:23<05:27, 446kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 719M/862M [05:25<04:07, 578kB/s].vector_cache/glove.6B.zip:  83%|████████▎ | 720M/862M [05:25<03:07, 760kB/s].vector_cache/glove.6B.zip:  84%|████████▎ | 721M/862M [05:25<02:13, 1.06MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<02:04, 1.12MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 723M/862M [05:27<01:55, 1.20MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 724M/862M [05:27<01:27, 1.58MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 727M/862M [05:29<01:21, 1.65MB/s].vector_cache/glove.6B.zip:  84%|████████▍ | 728M/862M [05:29<01:10, 1.90MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 729M/862M [05:29<00:51, 2.56MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<01:06, 1.96MB/s].vector_cache/glove.6B.zip:  85%|████████▍ | 732M/862M [05:31<00:59, 2.18MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 733M/862M [05:31<00:44, 2.88MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<01:00, 2.09MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 736M/862M [05:33<01:07, 1.86MB/s].vector_cache/glove.6B.zip:  85%|████████▌ | 737M/862M [05:33<00:52, 2.38MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 738M/862M [05:33<00:38, 3.23MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:35<01:13, 1.66MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 740M/862M [05:35<01:04, 1.90MB/s].vector_cache/glove.6B.zip:  86%|████████▌ | 742M/862M [05:35<00:47, 2.54MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<01:00, 1.96MB/s].vector_cache/glove.6B.zip:  86%|████████▋ | 744M/862M [05:37<00:54, 2.18MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 746M/862M [05:37<00:40, 2.89MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:39<00:54, 2.09MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 748M/862M [05:39<00:49, 2.29MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 750M/862M [05:39<00:36, 3.05MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:41<00:51, 2.14MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 752M/862M [05:41<00:47, 2.33MB/s].vector_cache/glove.6B.zip:  87%|████████▋ | 754M/862M [05:41<00:34, 3.10MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 756M/862M [05:43<00:49, 2.15MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 757M/862M [05:43<00:45, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 758M/862M [05:43<00:33, 3.12MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 760M/862M [05:45<00:47, 2.16MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 761M/862M [05:45<00:43, 2.34MB/s].vector_cache/glove.6B.zip:  88%|████████▊ | 762M/862M [05:45<00:32, 3.08MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 764M/862M [05:47<00:45, 2.15MB/s].vector_cache/glove.6B.zip:  89%|████████▊ | 765M/862M [05:47<00:51, 1.89MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 765M/862M [05:47<00:39, 2.43MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 767M/862M [05:47<00:28, 3.30MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:49<01:00, 1.54MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 769M/862M [05:49<00:51, 1.79MB/s].vector_cache/glove.6B.zip:  89%|████████▉ | 770M/862M [05:49<00:37, 2.43MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:51<00:46, 1.91MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 773M/862M [05:51<00:50, 1.75MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 774M/862M [05:51<00:39, 2.26MB/s].vector_cache/glove.6B.zip:  90%|████████▉ | 776M/862M [05:51<00:27, 3.10MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:52<01:09, 1.23MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 777M/862M [05:53<00:56, 1.49MB/s].vector_cache/glove.6B.zip:  90%|█████████ | 779M/862M [05:53<00:41, 2.02MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:54<00:47, 1.72MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 781M/862M [05:55<00:41, 1.96MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 783M/862M [05:55<00:29, 2.65MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:56<00:38, 1.99MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 785M/862M [05:57<00:42, 1.80MB/s].vector_cache/glove.6B.zip:  91%|█████████ | 786M/862M [05:57<00:33, 2.30MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:57<00:22, 3.18MB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<36:07, 33.7kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 789M/862M [05:58<25:17, 47.9kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 791M/862M [05:59<17:20, 68.3kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<12:02, 95.4kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 793M/862M [06:00<08:38, 133kB/s] .vector_cache/glove.6B.zip:  92%|█████████▏| 794M/862M [06:01<06:02, 188kB/s].vector_cache/glove.6B.zip:  92%|█████████▏| 797M/862M [06:02<04:12, 257kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 798M/862M [06:02<03:02, 354kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 799M/862M [06:03<02:06, 499kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 801M/862M [06:04<01:39, 611kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 802M/862M [06:04<01:14, 812kB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 803M/862M [06:04<00:52, 1.13MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:06<00:48, 1.17MB/s].vector_cache/glove.6B.zip:  93%|█████████▎| 806M/862M [06:06<00:39, 1.42MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 807M/862M [06:06<00:29, 1.90MB/s].vector_cache/glove.6B.zip:  94%|█████████▎| 808M/862M [06:07<00:21, 2.48MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 809M/862M [06:07<00:16, 3.13MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:07<00:14, 3.67MB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<43:37, 20.1kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<31:14, 28.0kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 810M/862M [06:08<21:58, 39.6kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:09<15:12, 56.5kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 811M/862M [06:09<10:31, 80.4kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 812M/862M [06:09<07:15, 114kB/s] .vector_cache/glove.6B.zip:  94%|█████████▍| 813M/862M [06:09<05:02, 162kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<03:54, 206kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<03:01, 265kB/s].vector_cache/glove.6B.zip:  94%|█████████▍| 814M/862M [06:10<02:10, 366kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 815M/862M [06:11<01:31, 513kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 817M/862M [06:11<01:03, 716kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:11<00:44, 989kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<04:55, 150kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 818M/862M [06:12<03:41, 199kB/s].vector_cache/glove.6B.zip:  95%|█████████▍| 819M/862M [06:12<02:36, 279kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 820M/862M [06:12<01:48, 392kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 821M/862M [06:13<01:14, 551kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:13<00:51, 771kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<33:59, 19.7kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<24:13, 27.5kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 822M/862M [06:14<16:57, 39.1kB/s].vector_cache/glove.6B.zip:  95%|█████████▌| 823M/862M [06:14<11:39, 55.7kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 824M/862M [06:15<07:54, 79.4kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:15<05:21, 113kB/s] .vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<04:18, 139kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 826M/862M [06:16<03:12, 186kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 827M/862M [06:16<02:15, 261kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 828M/862M [06:16<01:32, 368kB/s].vector_cache/glove.6B.zip:  96%|█████████▌| 829M/862M [06:17<01:03, 517kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<00:57, 555kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 830M/862M [06:18<00:59, 535kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 831M/862M [06:18<00:46, 682kB/s].vector_cache/glove.6B.zip:  96%|█████████▋| 832M/862M [06:18<00:32, 939kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 833M/862M [06:19<00:22, 1.29MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 834M/862M [06:19<00:15, 1.75MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:20<01:02, 441kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:20<00:52, 529kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 835M/862M [06:20<00:37, 720kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 836M/862M [06:20<00:26, 980kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 837M/862M [06:20<00:18, 1.36MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 838M/862M [06:21<00:13, 1.81MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:22, 1.04MB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:30, 778kB/s] .vector_cache/glove.6B.zip:  97%|█████████▋| 839M/862M [06:22<00:24, 944kB/s].vector_cache/glove.6B.zip:  97%|█████████▋| 840M/862M [06:22<00:17, 1.29MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 841M/862M [06:22<00:11, 1.76MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 842M/862M [06:23<00:08, 2.26MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:15, 1.23MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:16, 1.15MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 843M/862M [06:24<00:12, 1.46MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 845M/862M [06:24<00:08, 1.97MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 846M/862M [06:24<00:06, 2.62MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:25<00:04, 3.30MB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<01:20, 190kB/s] .vector_cache/glove.6B.zip:  98%|█████████▊| 847M/862M [06:26<01:00, 251kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 848M/862M [06:26<00:41, 349kB/s].vector_cache/glove.6B.zip:  98%|█████████▊| 849M/862M [06:26<00:27, 492kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 850M/862M [06:26<00:18, 685kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:27<00:11, 952kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:48, 231kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:40, 275kB/s].vector_cache/glove.6B.zip:  99%|█████████▊| 851M/862M [06:28<00:29, 369kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 852M/862M [06:28<00:19, 518kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 854M/862M [06:28<00:11, 724kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:29<00:07, 1.01MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:14, 490kB/s] .vector_cache/glove.6B.zip:  99%|█████████▉| 855M/862M [06:30<00:11, 577kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 856M/862M [06:30<00:08, 782kB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 857M/862M [06:30<00:04, 1.08MB/s].vector_cache/glove.6B.zip:  99%|█████████▉| 858M/862M [06:30<00:03, 1.46MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:30<00:01, 1.99MB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:32<00:04, 639kB/s] .vector_cache/glove.6B.zip: 100%|█████████▉| 859M/862M [06:32<00:04, 558kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 860M/862M [06:32<00:03, 711kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 861M/862M [06:32<00:01, 976kB/s].vector_cache/glove.6B.zip: 100%|█████████▉| 862M/862M [06:32<00:00, 1.33MB/s].vector_cache/glove.6B.zip: 862MB [06:32, 2.19MB/s]                           
  0%|          | 0/400000 [00:00<?, ?it/s]  0%|          | 201/400000 [00:00<03:19, 2008.83it/s]  0%|          | 1062/400000 [00:00<02:32, 2608.86it/s]  0%|          | 1939/400000 [00:00<02:00, 3305.48it/s]  1%|          | 2817/400000 [00:00<01:37, 4065.94it/s]  1%|          | 3647/400000 [00:00<01:22, 4800.23it/s]  1%|          | 4484/400000 [00:00<01:11, 5503.70it/s]  1%|▏         | 5384/400000 [00:00<01:03, 6229.09it/s]  2%|▏         | 6194/400000 [00:00<00:58, 6691.29it/s]  2%|▏         | 6977/400000 [00:00<00:56, 6947.98it/s]  2%|▏         | 7754/400000 [00:01<00:55, 7114.06it/s]  2%|▏         | 8524/400000 [00:01<00:53, 7270.74it/s]  2%|▏         | 9337/400000 [00:01<00:52, 7508.38it/s]  3%|▎         | 10120/400000 [00:01<00:51, 7508.79it/s]  3%|▎         | 10983/400000 [00:01<00:49, 7812.92it/s]  3%|▎         | 11853/400000 [00:01<00:48, 8058.87it/s]  3%|▎         | 12674/400000 [00:01<00:48, 7973.22it/s]  3%|▎         | 13532/400000 [00:01<00:47, 8143.80it/s]  4%|▎         | 14364/400000 [00:01<00:47, 8194.90it/s]  4%|▍         | 15194/400000 [00:01<00:46, 8224.09it/s]  4%|▍         | 16084/400000 [00:02<00:45, 8414.97it/s]  4%|▍         | 16930/400000 [00:02<00:46, 8270.93it/s]  4%|▍         | 17769/400000 [00:02<00:46, 8305.91it/s]  5%|▍         | 18609/400000 [00:02<00:45, 8331.36it/s]  5%|▍         | 19444/400000 [00:02<00:46, 8250.74it/s]  5%|▌         | 20271/400000 [00:02<00:47, 7975.13it/s]  5%|▌         | 21072/400000 [00:02<00:49, 7675.49it/s]  5%|▌         | 21872/400000 [00:02<00:48, 7769.45it/s]  6%|▌         | 22672/400000 [00:02<00:48, 7835.20it/s]  6%|▌         | 23470/400000 [00:02<00:47, 7876.66it/s]  6%|▌         | 24267/400000 [00:03<00:47, 7904.24it/s]  6%|▋         | 25059/400000 [00:03<00:47, 7820.69it/s]  6%|▋         | 25864/400000 [00:03<00:47, 7887.49it/s]  7%|▋         | 26663/400000 [00:03<00:47, 7914.99it/s]  7%|▋         | 27459/400000 [00:03<00:47, 7926.19it/s]  7%|▋         | 28287/400000 [00:03<00:46, 8025.82it/s]  7%|▋         | 29116/400000 [00:03<00:45, 8101.45it/s]  7%|▋         | 29929/400000 [00:03<00:45, 8107.85it/s]  8%|▊         | 30796/400000 [00:03<00:44, 8266.40it/s]  8%|▊         | 31704/400000 [00:03<00:43, 8492.96it/s]  8%|▊         | 32584/400000 [00:04<00:42, 8581.95it/s]  8%|▊         | 33445/400000 [00:04<00:42, 8539.16it/s]  9%|▊         | 34301/400000 [00:04<00:44, 8284.93it/s]  9%|▉         | 35136/400000 [00:04<00:43, 8303.70it/s]  9%|▉         | 36028/400000 [00:04<00:42, 8478.55it/s]  9%|▉         | 36901/400000 [00:04<00:42, 8550.90it/s]  9%|▉         | 37771/400000 [00:04<00:42, 8592.62it/s] 10%|▉         | 38632/400000 [00:04<00:42, 8461.96it/s] 10%|▉         | 39524/400000 [00:04<00:41, 8593.74it/s] 10%|█         | 40461/400000 [00:04<00:40, 8812.15it/s] 10%|█         | 41357/400000 [00:05<00:40, 8853.34it/s] 11%|█         | 42245/400000 [00:05<00:40, 8780.36it/s] 11%|█         | 43125/400000 [00:05<00:41, 8542.72it/s] 11%|█         | 43982/400000 [00:05<00:43, 8172.50it/s] 11%|█         | 44828/400000 [00:05<00:43, 8254.33it/s] 11%|█▏        | 45758/400000 [00:05<00:41, 8542.32it/s] 12%|█▏        | 46636/400000 [00:05<00:41, 8610.65it/s] 12%|█▏        | 47501/400000 [00:05<00:42, 8346.42it/s] 12%|█▏        | 48341/400000 [00:05<00:43, 8065.84it/s] 12%|█▏        | 49153/400000 [00:06<00:44, 7965.92it/s] 12%|█▏        | 49954/400000 [00:06<00:45, 7646.68it/s] 13%|█▎        | 50759/400000 [00:06<00:44, 7762.68it/s] 13%|█▎        | 51540/400000 [00:06<00:45, 7724.53it/s] 13%|█▎        | 52370/400000 [00:06<00:44, 7887.16it/s] 13%|█▎        | 53187/400000 [00:06<00:43, 7968.22it/s] 13%|█▎        | 53987/400000 [00:06<00:44, 7835.25it/s] 14%|█▎        | 54812/400000 [00:06<00:43, 7954.12it/s] 14%|█▍        | 55610/400000 [00:06<00:43, 7853.80it/s] 14%|█▍        | 56397/400000 [00:06<00:44, 7764.59it/s] 14%|█▍        | 57228/400000 [00:07<00:43, 7919.40it/s] 15%|█▍        | 58033/400000 [00:07<00:42, 7955.42it/s] 15%|█▍        | 58906/400000 [00:07<00:41, 8172.26it/s] 15%|█▍        | 59726/400000 [00:07<00:41, 8174.94it/s] 15%|█▌        | 60546/400000 [00:07<00:41, 8118.45it/s] 15%|█▌        | 61400/400000 [00:07<00:41, 8237.72it/s] 16%|█▌        | 62226/400000 [00:07<00:42, 8032.44it/s] 16%|█▌        | 63117/400000 [00:07<00:40, 8275.68it/s] 16%|█▌        | 63974/400000 [00:07<00:40, 8359.40it/s] 16%|█▌        | 64842/400000 [00:07<00:39, 8451.30it/s] 16%|█▋        | 65763/400000 [00:08<00:38, 8665.15it/s] 17%|█▋        | 66658/400000 [00:08<00:38, 8747.24it/s] 17%|█▋        | 67535/400000 [00:08<00:38, 8740.06it/s] 17%|█▋        | 68425/400000 [00:08<00:37, 8785.57it/s] 17%|█▋        | 69314/400000 [00:08<00:37, 8814.46it/s] 18%|█▊        | 70197/400000 [00:08<00:37, 8692.17it/s] 18%|█▊        | 71068/400000 [00:08<00:37, 8675.20it/s] 18%|█▊        | 71965/400000 [00:08<00:37, 8758.81it/s] 18%|█▊        | 72846/400000 [00:08<00:37, 8772.40it/s] 18%|█▊        | 73738/400000 [00:08<00:37, 8814.40it/s] 19%|█▊        | 74646/400000 [00:09<00:36, 8891.47it/s] 19%|█▉        | 75536/400000 [00:09<00:36, 8872.13it/s] 19%|█▉        | 76424/400000 [00:09<00:38, 8440.40it/s] 19%|█▉        | 77273/400000 [00:09<00:39, 8215.33it/s] 20%|█▉        | 78133/400000 [00:09<00:38, 8326.87it/s] 20%|█▉        | 78991/400000 [00:09<00:38, 8400.19it/s] 20%|█▉        | 79834/400000 [00:09<00:38, 8329.21it/s] 20%|██        | 80698/400000 [00:09<00:37, 8418.20it/s] 20%|██        | 81542/400000 [00:09<00:38, 8369.54it/s] 21%|██        | 82381/400000 [00:10<00:37, 8374.37it/s] 21%|██        | 83269/400000 [00:10<00:37, 8517.58it/s] 21%|██        | 84122/400000 [00:10<00:37, 8520.44it/s] 21%|██        | 84975/400000 [00:10<00:37, 8418.84it/s] 21%|██▏       | 85818/400000 [00:10<00:37, 8361.09it/s] 22%|██▏       | 86693/400000 [00:10<00:36, 8471.04it/s] 22%|██▏       | 87594/400000 [00:10<00:36, 8622.82it/s] 22%|██▏       | 88458/400000 [00:10<00:36, 8566.77it/s] 22%|██▏       | 89348/400000 [00:10<00:35, 8661.93it/s] 23%|██▎       | 90216/400000 [00:10<00:36, 8537.60it/s] 23%|██▎       | 91071/400000 [00:11<00:37, 8346.42it/s] 23%|██▎       | 91982/400000 [00:11<00:35, 8560.71it/s] 23%|██▎       | 92841/400000 [00:11<00:35, 8566.47it/s] 23%|██▎       | 93700/400000 [00:11<00:35, 8563.29it/s] 24%|██▎       | 94558/400000 [00:11<00:36, 8393.21it/s] 24%|██▍       | 95455/400000 [00:11<00:35, 8557.69it/s] 24%|██▍       | 96378/400000 [00:11<00:34, 8748.84it/s] 24%|██▍       | 97256/400000 [00:11<00:35, 8562.27it/s] 25%|██▍       | 98116/400000 [00:11<00:35, 8572.01it/s] 25%|██▍       | 98975/400000 [00:11<00:35, 8547.36it/s] 25%|██▍       | 99832/400000 [00:12<00:35, 8486.06it/s] 25%|██▌       | 100682/400000 [00:12<00:35, 8481.04it/s] 25%|██▌       | 101531/400000 [00:12<00:35, 8374.70it/s] 26%|██▌       | 102392/400000 [00:12<00:35, 8441.70it/s] 26%|██▌       | 103250/400000 [00:12<00:35, 8475.75it/s] 26%|██▌       | 104099/400000 [00:12<00:35, 8425.31it/s] 26%|██▌       | 104942/400000 [00:12<00:35, 8297.12it/s] 26%|██▋       | 105773/400000 [00:12<00:35, 8208.35it/s] 27%|██▋       | 106642/400000 [00:12<00:35, 8345.94it/s] 27%|██▋       | 107479/400000 [00:12<00:35, 8350.35it/s] 27%|██▋       | 108358/400000 [00:13<00:34, 8475.46it/s] 27%|██▋       | 109207/400000 [00:13<00:34, 8479.29it/s] 28%|██▊       | 110056/400000 [00:13<00:34, 8340.97it/s] 28%|██▊       | 110906/400000 [00:13<00:34, 8386.93it/s] 28%|██▊       | 111746/400000 [00:13<00:34, 8271.15it/s] 28%|██▊       | 112597/400000 [00:13<00:34, 8339.56it/s] 28%|██▊       | 113490/400000 [00:13<00:33, 8507.66it/s] 29%|██▊       | 114343/400000 [00:13<00:33, 8457.26it/s] 29%|██▉       | 115190/400000 [00:13<00:34, 8372.52it/s] 29%|██▉       | 116029/400000 [00:13<00:34, 8268.25it/s] 29%|██▉       | 116857/400000 [00:14<00:34, 8161.06it/s] 29%|██▉       | 117684/400000 [00:14<00:34, 8190.71it/s] 30%|██▉       | 118504/400000 [00:14<00:36, 7762.37it/s] 30%|██▉       | 119286/400000 [00:14<00:36, 7730.20it/s] 30%|███       | 120144/400000 [00:14<00:35, 7966.39it/s] 30%|███       | 120961/400000 [00:14<00:34, 8026.18it/s] 30%|███       | 121826/400000 [00:14<00:33, 8201.63it/s] 31%|███       | 122650/400000 [00:14<00:33, 8185.83it/s] 31%|███       | 123516/400000 [00:14<00:33, 8320.55it/s] 31%|███       | 124432/400000 [00:15<00:32, 8553.82it/s] 31%|███▏      | 125348/400000 [00:15<00:31, 8726.24it/s] 32%|███▏      | 126286/400000 [00:15<00:30, 8911.97it/s] 32%|███▏      | 127181/400000 [00:15<00:30, 8874.56it/s] 32%|███▏      | 128072/400000 [00:15<00:30, 8884.83it/s] 32%|███▏      | 128963/400000 [00:15<00:30, 8873.91it/s] 32%|███▏      | 129898/400000 [00:15<00:29, 9010.13it/s] 33%|███▎      | 130826/400000 [00:15<00:29, 9087.68it/s] 33%|███▎      | 131736/400000 [00:15<00:30, 8771.77it/s] 33%|███▎      | 132617/400000 [00:15<00:31, 8359.33it/s] 33%|███▎      | 133460/400000 [00:16<00:32, 8212.51it/s] 34%|███▎      | 134293/400000 [00:16<00:32, 8245.71it/s] 34%|███▍      | 135122/400000 [00:16<00:32, 8249.48it/s] 34%|███▍      | 135975/400000 [00:16<00:31, 8330.23it/s] 34%|███▍      | 136901/400000 [00:16<00:30, 8586.86it/s] 34%|███▍      | 137783/400000 [00:16<00:30, 8654.54it/s] 35%|███▍      | 138700/400000 [00:16<00:29, 8801.99it/s] 35%|███▍      | 139618/400000 [00:16<00:29, 8911.78it/s] 35%|███▌      | 140512/400000 [00:16<00:29, 8773.00it/s] 35%|███▌      | 141392/400000 [00:16<00:30, 8525.97it/s] 36%|███▌      | 142267/400000 [00:17<00:30, 8589.74it/s] 36%|███▌      | 143149/400000 [00:17<00:29, 8656.95it/s] 36%|███▌      | 144017/400000 [00:17<00:30, 8513.73it/s] 36%|███▌      | 144871/400000 [00:17<00:30, 8308.75it/s] 36%|███▋      | 145705/400000 [00:17<00:30, 8275.10it/s] 37%|███▋      | 146535/400000 [00:17<00:31, 7976.68it/s] 37%|███▋      | 147337/400000 [00:17<00:31, 7922.44it/s] 37%|███▋      | 148159/400000 [00:17<00:31, 8008.30it/s] 37%|███▋      | 148962/400000 [00:17<00:32, 7759.09it/s] 37%|███▋      | 149741/400000 [00:18<00:32, 7672.37it/s] 38%|███▊      | 150545/400000 [00:18<00:32, 7778.34it/s] 38%|███▊      | 151370/400000 [00:18<00:31, 7911.69it/s] 38%|███▊      | 152208/400000 [00:18<00:30, 8044.56it/s] 38%|███▊      | 153049/400000 [00:18<00:30, 8148.44it/s] 38%|███▊      | 153966/400000 [00:18<00:29, 8429.43it/s] 39%|███▊      | 154821/400000 [00:18<00:28, 8464.81it/s] 39%|███▉      | 155755/400000 [00:18<00:28, 8706.76it/s] 39%|███▉      | 156649/400000 [00:18<00:27, 8774.22it/s] 39%|███▉      | 157530/400000 [00:18<00:28, 8610.25it/s] 40%|███▉      | 158452/400000 [00:19<00:27, 8783.83it/s] 40%|███▉      | 159333/400000 [00:19<00:27, 8788.78it/s] 40%|████      | 160227/400000 [00:19<00:27, 8830.71it/s] 40%|████      | 161112/400000 [00:19<00:27, 8534.23it/s] 40%|████      | 161973/400000 [00:19<00:27, 8556.33it/s] 41%|████      | 162867/400000 [00:19<00:27, 8667.60it/s] 41%|████      | 163736/400000 [00:19<00:27, 8612.17it/s] 41%|████      | 164682/400000 [00:19<00:26, 8847.89it/s] 41%|████▏     | 165570/400000 [00:19<00:26, 8806.33it/s] 42%|████▏     | 166475/400000 [00:19<00:26, 8877.97it/s] 42%|████▏     | 167399/400000 [00:20<00:25, 8981.97it/s] 42%|████▏     | 168302/400000 [00:20<00:25, 8993.79it/s] 42%|████▏     | 169240/400000 [00:20<00:25, 9104.78it/s] 43%|████▎     | 170152/400000 [00:20<00:25, 9105.10it/s] 43%|████▎     | 171064/400000 [00:20<00:25, 9091.90it/s] 43%|████▎     | 171978/400000 [00:20<00:25, 9105.89it/s] 43%|████▎     | 172902/400000 [00:20<00:24, 9143.50it/s] 43%|████▎     | 173841/400000 [00:20<00:24, 9215.85it/s] 44%|████▎     | 174763/400000 [00:20<00:24, 9083.44it/s] 44%|████▍     | 175673/400000 [00:20<00:25, 8764.70it/s] 44%|████▍     | 176553/400000 [00:21<00:25, 8649.31it/s] 44%|████▍     | 177421/400000 [00:21<00:25, 8595.55it/s] 45%|████▍     | 178283/400000 [00:21<00:25, 8582.99it/s] 45%|████▍     | 179183/400000 [00:21<00:25, 8702.59it/s] 45%|████▌     | 180063/400000 [00:21<00:25, 8730.64it/s] 45%|████▌     | 180937/400000 [00:21<00:25, 8692.53it/s] 45%|████▌     | 181850/400000 [00:21<00:24, 8818.81it/s] 46%|████▌     | 182782/400000 [00:21<00:24, 8962.86it/s] 46%|████▌     | 183680/400000 [00:21<00:24, 8875.05it/s] 46%|████▌     | 184569/400000 [00:21<00:24, 8822.16it/s] 46%|████▋     | 185475/400000 [00:22<00:24, 8889.84it/s] 47%|████▋     | 186392/400000 [00:22<00:23, 8969.57it/s] 47%|████▋     | 187290/400000 [00:22<00:23, 8889.20it/s] 47%|████▋     | 188180/400000 [00:22<00:23, 8837.15it/s] 47%|████▋     | 189102/400000 [00:22<00:23, 8948.08it/s] 47%|████▋     | 189998/400000 [00:22<00:23, 8873.17it/s] 48%|████▊     | 190886/400000 [00:22<00:24, 8619.83it/s] 48%|████▊     | 191822/400000 [00:22<00:23, 8828.68it/s] 48%|████▊     | 192708/400000 [00:22<00:23, 8757.83it/s] 48%|████▊     | 193614/400000 [00:23<00:23, 8845.06it/s] 49%|████▊     | 194501/400000 [00:23<00:23, 8789.55it/s] 49%|████▉     | 195382/400000 [00:23<00:23, 8718.80it/s] 49%|████▉     | 196287/400000 [00:23<00:23, 8812.71it/s] 49%|████▉     | 197170/400000 [00:23<00:23, 8677.12it/s] 50%|████▉     | 198039/400000 [00:23<00:23, 8676.48it/s] 50%|████▉     | 198927/400000 [00:23<00:23, 8735.77it/s] 50%|████▉     | 199815/400000 [00:23<00:22, 8777.65it/s] 50%|█████     | 200759/400000 [00:23<00:22, 8965.21it/s] 50%|█████     | 201657/400000 [00:23<00:22, 8879.95it/s] 51%|█████     | 202571/400000 [00:24<00:22, 8954.82it/s] 51%|█████     | 203468/400000 [00:24<00:22, 8904.82it/s] 51%|█████     | 204360/400000 [00:24<00:22, 8832.16it/s] 51%|█████▏    | 205244/400000 [00:24<00:22, 8562.88it/s] 52%|█████▏    | 206103/400000 [00:24<00:23, 8306.66it/s] 52%|█████▏    | 206982/400000 [00:24<00:22, 8443.87it/s] 52%|█████▏    | 207830/400000 [00:24<00:22, 8441.74it/s] 52%|█████▏    | 208693/400000 [00:24<00:22, 8495.34it/s] 52%|█████▏    | 209590/400000 [00:24<00:22, 8631.95it/s] 53%|█████▎    | 210455/400000 [00:24<00:22, 8375.66it/s] 53%|█████▎    | 211296/400000 [00:25<00:22, 8309.99it/s] 53%|█████▎    | 212130/400000 [00:25<00:23, 8125.35it/s] 53%|█████▎    | 212945/400000 [00:25<00:23, 8114.84it/s] 53%|█████▎    | 213759/400000 [00:25<00:23, 7836.29it/s] 54%|█████▎    | 214557/400000 [00:25<00:23, 7878.61it/s] 54%|█████▍    | 215348/400000 [00:25<00:23, 7860.55it/s] 54%|█████▍    | 216194/400000 [00:25<00:22, 8029.93it/s] 54%|█████▍    | 217060/400000 [00:25<00:22, 8205.49it/s] 54%|█████▍    | 217943/400000 [00:25<00:21, 8381.65it/s] 55%|█████▍    | 218784/400000 [00:25<00:22, 8093.40it/s] 55%|█████▍    | 219634/400000 [00:26<00:22, 8043.00it/s] 55%|█████▌    | 220452/400000 [00:26<00:22, 8081.57it/s] 55%|█████▌    | 221326/400000 [00:26<00:21, 8267.98it/s] 56%|█████▌    | 222294/400000 [00:26<00:20, 8646.35it/s] 56%|█████▌    | 223165/400000 [00:26<00:20, 8587.33it/s] 56%|█████▌    | 224096/400000 [00:26<00:20, 8790.32it/s] 56%|█████▌    | 224982/400000 [00:26<00:19, 8809.26it/s] 56%|█████▋    | 225903/400000 [00:26<00:19, 8924.96it/s] 57%|█████▋    | 226801/400000 [00:26<00:19, 8940.88it/s] 57%|█████▋    | 227697/400000 [00:27<00:19, 8735.92it/s] 57%|█████▋    | 228611/400000 [00:27<00:19, 8850.76it/s] 57%|█████▋    | 229558/400000 [00:27<00:18, 9025.76it/s] 58%|█████▊    | 230463/400000 [00:27<00:18, 8992.22it/s] 58%|█████▊    | 231428/400000 [00:27<00:18, 9177.95it/s] 58%|█████▊    | 232348/400000 [00:27<00:18, 8921.30it/s] 58%|█████▊    | 233244/400000 [00:27<00:19, 8632.12it/s] 59%|█████▊    | 234112/400000 [00:27<00:20, 8209.48it/s] 59%|█████▊    | 234941/400000 [00:27<00:20, 8119.43it/s] 59%|█████▉    | 235759/400000 [00:27<00:20, 8014.66it/s] 59%|█████▉    | 236591/400000 [00:28<00:20, 8102.29it/s] 59%|█████▉    | 237497/400000 [00:28<00:19, 8366.44it/s] 60%|█████▉    | 238363/400000 [00:28<00:19, 8451.97it/s] 60%|█████▉    | 239212/400000 [00:28<00:19, 8363.35it/s] 60%|██████    | 240051/400000 [00:28<00:19, 8077.45it/s] 60%|██████    | 240863/400000 [00:28<00:20, 7945.93it/s] 60%|██████    | 241679/400000 [00:28<00:19, 8006.52it/s] 61%|██████    | 242482/400000 [00:28<00:19, 7935.64it/s] 61%|██████    | 243278/400000 [00:28<00:19, 7880.35it/s] 61%|██████    | 244068/400000 [00:28<00:19, 7799.05it/s] 61%|██████    | 244855/400000 [00:29<00:19, 7819.32it/s] 61%|██████▏   | 245643/400000 [00:29<00:19, 7837.13it/s] 62%|██████▏   | 246428/400000 [00:29<00:20, 7670.89it/s] 62%|██████▏   | 247197/400000 [00:29<00:19, 7670.26it/s] 62%|██████▏   | 247965/400000 [00:29<00:20, 7576.23it/s] 62%|██████▏   | 248724/400000 [00:29<00:20, 7536.49it/s] 62%|██████▏   | 249479/400000 [00:29<00:19, 7540.33it/s] 63%|██████▎   | 250258/400000 [00:29<00:19, 7611.56it/s] 63%|██████▎   | 251041/400000 [00:29<00:19, 7674.96it/s] 63%|██████▎   | 251832/400000 [00:30<00:19, 7742.71it/s] 63%|██████▎   | 252711/400000 [00:30<00:18, 8028.02it/s] 63%|██████▎   | 253602/400000 [00:30<00:17, 8273.03it/s] 64%|██████▎   | 254473/400000 [00:30<00:17, 8398.61it/s] 64%|██████▍   | 255393/400000 [00:30<00:16, 8622.98it/s] 64%|██████▍   | 256278/400000 [00:30<00:16, 8689.80it/s] 64%|██████▍   | 257181/400000 [00:30<00:16, 8789.01it/s] 65%|██████▍   | 258074/400000 [00:30<00:16, 8829.63it/s] 65%|██████▍   | 258959/400000 [00:30<00:15, 8823.04it/s] 65%|██████▍   | 259864/400000 [00:30<00:15, 8887.18it/s] 65%|██████▌   | 260754/400000 [00:31<00:16, 8634.72it/s] 65%|██████▌   | 261620/400000 [00:31<00:16, 8529.30it/s] 66%|██████▌   | 262475/400000 [00:31<00:16, 8466.71it/s] 66%|██████▌   | 263343/400000 [00:31<00:16, 8529.27it/s] 66%|██████▌   | 264253/400000 [00:31<00:15, 8690.41it/s] 66%|██████▋   | 265124/400000 [00:31<00:15, 8557.59it/s] 66%|██████▋   | 265989/400000 [00:31<00:15, 8582.60it/s] 67%|██████▋   | 266890/400000 [00:31<00:15, 8706.49it/s] 67%|██████▋   | 267803/400000 [00:31<00:14, 8828.34it/s] 67%|██████▋   | 268738/400000 [00:31<00:14, 8977.06it/s] 67%|██████▋   | 269638/400000 [00:32<00:14, 8981.86it/s] 68%|██████▊   | 270562/400000 [00:32<00:14, 9055.65it/s] 68%|██████▊   | 271519/400000 [00:32<00:13, 9203.12it/s] 68%|██████▊   | 272441/400000 [00:32<00:13, 9163.65it/s] 68%|██████▊   | 273397/400000 [00:32<00:13, 9278.81it/s] 69%|██████▊   | 274326/400000 [00:32<00:13, 9161.30it/s] 69%|██████▉   | 275244/400000 [00:32<00:14, 8906.71it/s] 69%|██████▉   | 276137/400000 [00:32<00:14, 8766.56it/s] 69%|██████▉   | 277019/400000 [00:32<00:14, 8779.96it/s] 69%|██████▉   | 277912/400000 [00:32<00:13, 8821.93it/s] 70%|██████▉   | 278796/400000 [00:33<00:13, 8686.61it/s] 70%|██████▉   | 279707/400000 [00:33<00:13, 8809.25it/s] 70%|███████   | 280637/400000 [00:33<00:13, 8948.52it/s] 70%|███████   | 281534/400000 [00:33<00:13, 8903.04it/s] 71%|███████   | 282449/400000 [00:33<00:13, 8974.40it/s] 71%|███████   | 283348/400000 [00:33<00:13, 8462.53it/s] 71%|███████   | 284201/400000 [00:33<00:13, 8352.53it/s] 71%|███████▏  | 285102/400000 [00:33<00:13, 8537.62it/s] 71%|███████▏  | 285989/400000 [00:33<00:13, 8633.59it/s] 72%|███████▏  | 286884/400000 [00:33<00:12, 8725.55it/s] 72%|███████▏  | 287760/400000 [00:34<00:13, 8437.95it/s] 72%|███████▏  | 288608/400000 [00:34<00:13, 8364.83it/s] 72%|███████▏  | 289448/400000 [00:34<00:13, 8320.34it/s] 73%|███████▎  | 290318/400000 [00:34<00:13, 8430.63it/s] 73%|███████▎  | 291249/400000 [00:34<00:12, 8674.68it/s] 73%|███████▎  | 292120/400000 [00:34<00:12, 8604.79it/s] 73%|███████▎  | 293059/400000 [00:34<00:12, 8823.69it/s] 73%|███████▎  | 293980/400000 [00:34<00:11, 8934.56it/s] 74%|███████▎  | 294876/400000 [00:34<00:11, 8899.66it/s] 74%|███████▍  | 295768/400000 [00:35<00:11, 8858.72it/s] 74%|███████▍  | 296656/400000 [00:35<00:11, 8718.98it/s] 74%|███████▍  | 297530/400000 [00:35<00:11, 8579.48it/s] 75%|███████▍  | 298390/400000 [00:35<00:11, 8503.69it/s] 75%|███████▍  | 299242/400000 [00:35<00:12, 8353.72it/s] 75%|███████▌  | 300079/400000 [00:35<00:12, 8248.56it/s] 75%|███████▌  | 300906/400000 [00:35<00:12, 8131.44it/s] 75%|███████▌  | 301748/400000 [00:35<00:11, 8215.67it/s] 76%|███████▌  | 302619/400000 [00:35<00:11, 8357.52it/s] 76%|███████▌  | 303457/400000 [00:35<00:11, 8272.87it/s] 76%|███████▌  | 304286/400000 [00:36<00:11, 8057.32it/s] 76%|███████▋  | 305094/400000 [00:36<00:12, 7672.70it/s] 76%|███████▋  | 305867/400000 [00:36<00:12, 7684.19it/s] 77%|███████▋  | 306670/400000 [00:36<00:11, 7781.78it/s] 77%|███████▋  | 307504/400000 [00:36<00:11, 7940.74it/s] 77%|███████▋  | 308315/400000 [00:36<00:11, 7990.00it/s] 77%|███████▋  | 309117/400000 [00:36<00:11, 7881.05it/s] 77%|███████▋  | 309939/400000 [00:36<00:11, 7979.32it/s] 78%|███████▊  | 310789/400000 [00:36<00:10, 8128.57it/s] 78%|███████▊  | 311630/400000 [00:36<00:10, 8210.15it/s] 78%|███████▊  | 312453/400000 [00:37<00:10, 8167.27it/s] 78%|███████▊  | 313271/400000 [00:37<00:10, 8066.52it/s] 79%|███████▊  | 314130/400000 [00:37<00:10, 8215.27it/s] 79%|███████▊  | 314994/400000 [00:37<00:10, 8335.87it/s] 79%|███████▉  | 315829/400000 [00:37<00:10, 8265.82it/s] 79%|███████▉  | 316657/400000 [00:37<00:10, 8215.33it/s] 79%|███████▉  | 317480/400000 [00:37<00:10, 7914.36it/s] 80%|███████▉  | 318295/400000 [00:37<00:10, 7981.57it/s] 80%|███████▉  | 319207/400000 [00:37<00:09, 8291.48it/s] 80%|████████  | 320073/400000 [00:38<00:09, 8396.63it/s] 80%|████████  | 320939/400000 [00:38<00:09, 8472.82it/s] 80%|████████  | 321789/400000 [00:38<00:09, 8345.45it/s] 81%|████████  | 322626/400000 [00:38<00:09, 8051.89it/s] 81%|████████  | 323436/400000 [00:38<00:09, 8058.70it/s] 81%|████████  | 324285/400000 [00:38<00:09, 8181.42it/s] 81%|████████▏ | 325138/400000 [00:38<00:09, 8281.92it/s] 81%|████████▏ | 325969/400000 [00:38<00:09, 8162.12it/s] 82%|████████▏ | 326827/400000 [00:38<00:08, 8281.19it/s] 82%|████████▏ | 327657/400000 [00:38<00:08, 8275.35it/s] 82%|████████▏ | 328522/400000 [00:39<00:08, 8383.92it/s] 82%|████████▏ | 329415/400000 [00:39<00:08, 8539.19it/s] 83%|████████▎ | 330271/400000 [00:39<00:08, 8377.57it/s] 83%|████████▎ | 331111/400000 [00:39<00:08, 8264.41it/s] 83%|████████▎ | 332000/400000 [00:39<00:08, 8441.11it/s] 83%|████████▎ | 332887/400000 [00:39<00:07, 8564.06it/s] 83%|████████▎ | 333746/400000 [00:39<00:07, 8417.89it/s] 84%|████████▎ | 334590/400000 [00:39<00:07, 8233.40it/s] 84%|████████▍ | 335416/400000 [00:39<00:07, 8147.44it/s] 84%|████████▍ | 336242/400000 [00:39<00:07, 8180.46it/s] 84%|████████▍ | 337062/400000 [00:40<00:07, 8114.71it/s] 84%|████████▍ | 337875/400000 [00:40<00:07, 8042.75it/s] 85%|████████▍ | 338681/400000 [00:40<00:07, 7947.55it/s] 85%|████████▍ | 339522/400000 [00:40<00:07, 8078.87it/s] 85%|████████▌ | 340346/400000 [00:40<00:07, 8125.45it/s] 85%|████████▌ | 341172/400000 [00:40<00:07, 8165.07it/s] 85%|████████▌ | 341990/400000 [00:40<00:07, 8089.74it/s] 86%|████████▌ | 342840/400000 [00:40<00:06, 8206.77it/s] 86%|████████▌ | 343716/400000 [00:40<00:06, 8364.11it/s] 86%|████████▌ | 344554/400000 [00:40<00:06, 8175.15it/s] 86%|████████▋ | 345437/400000 [00:41<00:06, 8360.34it/s] 87%|████████▋ | 346315/400000 [00:41<00:06, 8480.64it/s] 87%|████████▋ | 347166/400000 [00:41<00:06, 8453.04it/s] 87%|████████▋ | 348104/400000 [00:41<00:05, 8709.83it/s] 87%|████████▋ | 349003/400000 [00:41<00:05, 8790.05it/s] 87%|████████▋ | 349946/400000 [00:41<00:05, 8970.52it/s] 88%|████████▊ | 350846/400000 [00:41<00:05, 8961.68it/s] 88%|████████▊ | 351744/400000 [00:41<00:05, 8880.98it/s] 88%|████████▊ | 352661/400000 [00:41<00:05, 8963.47it/s] 88%|████████▊ | 353561/400000 [00:41<00:05, 8971.98it/s] 89%|████████▊ | 354469/400000 [00:42<00:05, 9003.35it/s] 89%|████████▉ | 355370/400000 [00:42<00:05, 8920.67it/s] 89%|████████▉ | 356263/400000 [00:42<00:04, 8872.62it/s] 89%|████████▉ | 357215/400000 [00:42<00:04, 9055.00it/s] 90%|████████▉ | 358122/400000 [00:42<00:04, 9009.94it/s] 90%|████████▉ | 359024/400000 [00:42<00:04, 8841.88it/s] 90%|████████▉ | 359910/400000 [00:42<00:04, 8176.12it/s] 90%|█████████ | 360789/400000 [00:42<00:04, 8350.35it/s] 90%|█████████ | 361633/400000 [00:42<00:04, 8370.90it/s] 91%|█████████ | 362477/400000 [00:43<00:04, 8389.71it/s] 91%|█████████ | 363380/400000 [00:43<00:04, 8570.72it/s] 91%|█████████ | 364241/400000 [00:43<00:04, 8357.05it/s] 91%|█████████▏| 365108/400000 [00:43<00:04, 8448.51it/s] 92%|█████████▏| 366060/400000 [00:43<00:03, 8743.14it/s] 92%|█████████▏| 366940/400000 [00:43<00:03, 8660.03it/s] 92%|█████████▏| 367850/400000 [00:43<00:03, 8786.83it/s] 92%|█████████▏| 368732/400000 [00:43<00:03, 8653.49it/s] 92%|█████████▏| 369617/400000 [00:43<00:03, 8711.12it/s] 93%|█████████▎| 370522/400000 [00:43<00:03, 8808.80it/s] 93%|█████████▎| 371407/400000 [00:44<00:03, 8820.68it/s] 93%|█████████▎| 372291/400000 [00:44<00:03, 8810.90it/s] 93%|█████████▎| 373173/400000 [00:44<00:03, 8504.56it/s] 94%|█████████▎| 374027/400000 [00:44<00:03, 8213.53it/s] 94%|█████████▎| 374950/400000 [00:44<00:02, 8492.64it/s] 94%|█████████▍| 375862/400000 [00:44<00:02, 8669.94it/s] 94%|█████████▍| 376773/400000 [00:44<00:02, 8797.10it/s] 94%|█████████▍| 377657/400000 [00:44<00:02, 8807.02it/s] 95%|█████████▍| 378587/400000 [00:44<00:02, 8948.84it/s] 95%|█████████▍| 379486/400000 [00:44<00:02, 8960.51it/s] 95%|█████████▌| 380384/400000 [00:45<00:02, 8927.13it/s] 95%|█████████▌| 381279/400000 [00:45<00:02, 8933.77it/s] 96%|█████████▌| 382174/400000 [00:45<00:02, 8852.32it/s] 96%|█████████▌| 383125/400000 [00:45<00:01, 9038.78it/s] 96%|█████████▌| 384031/400000 [00:45<00:01, 8969.35it/s] 96%|█████████▌| 384948/400000 [00:45<00:01, 9028.47it/s] 96%|█████████▋| 385852/400000 [00:45<00:01, 8798.55it/s] 97%|█████████▋| 386734/400000 [00:45<00:01, 8768.82it/s] 97%|█████████▋| 387621/400000 [00:45<00:01, 8798.37it/s] 97%|█████████▋| 388502/400000 [00:45<00:01, 8605.38it/s] 97%|█████████▋| 389365/400000 [00:46<00:01, 8268.55it/s] 98%|█████████▊| 390196/400000 [00:46<00:01, 7851.39it/s] 98%|█████████▊| 391066/400000 [00:46<00:01, 8087.51it/s] 98%|█████████▊| 391962/400000 [00:46<00:00, 8330.77it/s] 98%|█████████▊| 392886/400000 [00:46<00:00, 8582.00it/s] 98%|█████████▊| 393751/400000 [00:46<00:00, 8564.81it/s] 99%|█████████▊| 394613/400000 [00:46<00:00, 8302.07it/s] 99%|█████████▉| 395449/400000 [00:46<00:00, 8274.33it/s] 99%|█████████▉| 396280/400000 [00:46<00:00, 8176.66it/s] 99%|█████████▉| 397101/400000 [00:47<00:00, 8011.04it/s] 99%|█████████▉| 397905/400000 [00:47<00:00, 7676.12it/s]100%|█████████▉| 398678/400000 [00:47<00:00, 7556.70it/s]100%|█████████▉| 399438/400000 [00:47<00:00, 7524.51it/s]100%|█████████▉| 399999/400000 [00:47<00:00, 8430.78it/s]Preprocessing the text...
Creating tabular datasets...It might take a while to finish!
Building vocaulary...

  
 #####  get_Data DataLoader  

  ((<torchtext.data.dataset.TabularDataset object at 0x7f50933186d8>, <torchtext.data.dataset.TabularDataset object at 0x7f5093318828>, <torchtext.vocab.Vocab object at 0x7f5093318748>), {}) 


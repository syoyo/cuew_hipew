python gen.py cudart.json cudart-config.json
python gen.py cublas.json cublas-config.json
python gen.py cufft.json cufft-config.json
python gen.py curand.json curand-config.json
python gen.py nvjpeg.json nvjpeg-config.json
python gen.py cusparse.json cusparse-config.json

# depends on cusparse, cublas
python gen.py cusolver.json cusolver-config.json

FROM storyteller123/cuda118_cudnn_pytorch210_simpletransformer
COPY . /deploy_Mood/

WORKDIR /deploy_Mood/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt
RUN python3 -m pip install gdown
RUN gdown --folder 'https://drive.google.com/drive/folders/1NLziHG4yfSc5DX7d2MxtJj3pWuMJ3OiW?usp=sharing' -O /deploy_Mood/Mood/
# RUN gdown --folder 'https://drive.google.com/drive/folders/1Oez7sEnwvsb7g8qYNSXQ9lAMTEFFAkv0?usp=sharing' -O /deploy_Mood/Mood/


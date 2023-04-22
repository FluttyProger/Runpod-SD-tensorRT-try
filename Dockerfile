# Must use a Cuda version 11+
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git
RUN conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

RUN python3 -m pip install --upgrade tensorrt
RUN python3 -m pip install --upgrade polygraphy onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com
RUN python3 -m pip install onnxruntime

# We add the banana boilerplate here
ADD server.py .

# Define model used
ARG MODEL_NAME
ENV MODEL_NAME=XpucT/Deliberate

RUN git clone https://github.com/FluttyProger/diffusers

WORKDIR /diffusers

RUN pip install .

WORKDIR /

# Add your model weight files 
ADD download.py .
RUN python3 download.py

ADD app.py .

RUN wget -O /opt/conda/lib/python3.10/site-packages/torch/onnx/_constants.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/_constants.py

RUN wget -O /opt/conda/lib/python3.10/site-packages/torch/onnx/symbolic_opset14.py https://raw.githubusercontent.com/pytorch/pytorch/d06d195bcd960f530f8f0d5a1992ed68d2823d4e/torch/onnx/symbolic_opset14.py

EXPOSE 8000

CMD python3 -u server.py

FROM intel/ai-workflows:torch-2.2.0-huggingface-multinode-py3.10

WORKDIR /workspace/scripts

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY finetune.py /workspace/scripts/finetune.py

CMD ["/bin/bash"]
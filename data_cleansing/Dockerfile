FROM gcr.io/deeplearning-platform-release/sklearn-cpu:latest

WORKDIR /

RUN pip install cloudml-hypertune google-cloud-storage numpy pandas scikit-learn

COPY src/ /

ENTRYPOINT ["python", "main.py"]
FROM python:3.10-slim
COPY . /procting
WORKDIR /procting

RUN apt-get update && apt-get install -y \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly \
    libssl-dev
RUN apt-get install openssl
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8080

CMD ["python", "testt.py"]
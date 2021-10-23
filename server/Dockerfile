FROM openvino/ubuntu18_dev:latest
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD [ "python", "manage.py", "runserver" ]
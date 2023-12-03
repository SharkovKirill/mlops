Сервис позволяет обучать пока что два типа моделей (CatBoostClassifier, RandomForestClassifier), делать предсказания на них, сохранять и удалять их. Пользователь может задавать свои гиперпараметры, которые также сохраняются. Данные для обучения и предсказаний  передаются через JSON (REST).
Модели хранятся на s3 и отслеживаются в clearml,  то есть можно будет вернуться к ним даже после перезапуска. В grpc_client run есть пример работы с сервисом через gRPC(запускаеся на порту 50051). 

REST запускается на порту 8005, clearml:8080, mini0:9000

Есть 3 варианта запуска:
1. Вручную (streamlit пока только здесь)
2. Docker run (образ на docker hub)
3. Docker-compose up

На машине уже должен должен быть сконфигурированный clearml (https://clear.ml/docs/latest/docs/deploying_clearml/clearml_server_linux_mac/).
Все порты и адреса по умолчанию дефолтные, но лучше прокинуть их если что-то не так.

## Вручную
Задать переменные окржуения для подключения к minio если они не дефолтные.По дефолту стоит так:

MINIO_USER = os.environ.get("MINIO_USER", "user_login")
MINIO_PASSWORD = os.environ.get("MINIO_PASSWORD", "user_password")
MINIO_URL = os.environ.get("MINIO_URL", "http://localhost:9000")

Далее при активированном окружении:
>**для запуска с REST**: 
`poetry run start`
**для запуска с gRPC**: 
`poetry run python grpc/grpc_server.py`\
`poetry run python grpc/grpc_client.py `
**для запуска streamlit (после poetry run start_fast_api)**:
`streamlit run streamlit_app/streamlit_app.py`

## Docker run (тут тоже 2 варианта)
В обоих вариантах обязательно нужно прокидывать CLEARML_API_ACCESS_KEY,CLEARML_API_SECRET_KEY.
Также, прокинуть  MINIO_USER, MINIO_PASSWORD (скорее всего они не как у меня).
`docker run --name app -p 8005:8005 kilaef/mlops_hse:0.1.0`
**1. Закинуть minio, весь clearml, app в одну сеть (названия сервисов именно такие)**
>`docker network create ml_net`
`docker network connect ml_net app`
`docker network connect ml_net minio`
`docker network connect ml_net clearml-webserver`
...
`docker network connect ml_net clearml-apiserver`

**2. Не закидывать в одну сеть, но при запуске проеинуть еще CLEARML_API_HOST, CLEARML_WEB_HOST, CLEARML_FILES_HOST, MINIO_URL**

## Docker-compose up
Сразу запускается app, minio, clearml. Нужно прокинуть только  CLEARML_API_ACCESS_KEY, CLEARML_API_SECRET_KEY
`docker-compose up`



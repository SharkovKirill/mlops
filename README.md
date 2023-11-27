**для запуска с REST**: 
poetry run start

**для запуска с gRPC**: 
poetry run python grpc/grpc_server.py\
poetry run python grpc/grpc_client.py 

Сервис позволяет обучать два типа моделей (CatBoostClassifier, RandomForestClassifier), делать предсказания на них, сохранять и удалять их. Пользователь может задавать свои гиперпараметры, которые также сохраняются. Данные для обучения и предсказаний  передаются через JSON (REST).
Модели хранятся на диске, то есть можно будет вернуться к ним даже после перезапуска. В grpc_client run есть пример работы с сервисом через gRPC(запускаеся на порту 50051). REST запускается на порту 8005.

**Для запуска streamlit (после poetry run start_fast_api)**:
streamlit run streamlit_app/streamlit_app.py
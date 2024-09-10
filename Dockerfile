FROM python:3

ENV PYTHONUNBUFFERED True

EXPOSE 8080

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

COPY .streamlit/ $APP_HOME/.streamlit/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD streamlit run --server.port 8080 --server.enableCORS true app.py
FROM python:3.12

ENV PYTHONUNBUFFERED True

EXPOSE 8080

ENV APP_HOME /app

WORKDIR $APP_HOME

COPY . ./

COPY .streamlit/ ./.streamlit/

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

CMD streamlit run app.py --server.port 8080 --server.enableCORS true
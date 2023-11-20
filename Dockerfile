# Use an official Python runtime as a parent image
FROM python:3.11.4

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY ./requirements.txt /app/requirements.txt
COPY ./app.py /app/app.py
COPY ./data_daily.csv /app/data_daily.csv
COPY ./ty_custom_model.h5 /app/ty_custom_model.h5

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8501

# # Define environment variable
# ENV NAME World
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run app.py when the container launches
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
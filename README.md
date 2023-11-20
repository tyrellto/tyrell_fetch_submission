---

# Tyrell Fetch Submission

Provide a brief description of your project here.

## Getting Started

These instructions will guide you in getting a copy of the project running on your local machine for development and testing purposes.

### Prerequisites

What you need to install:

- **Git**
- **Docker**

#### Installing Git

Download and install Git from [Git's website](https://git-scm.com/downloads). Follow the installation instructions for your operating system.

#### Installing Docker

Download Docker Desktop from [Docker's website](https://www.docker.com/products/docker-desktop) and follow the installation instructions for your operating system.

### Installing and Running the Application

Follow these steps to set up your development environment.

#### Cloning the Repository

Open a terminal and run:

```bash
git clone https://github.com/tyrellto/tyrell_fetch_submission.git
cd tyrell_fetch_submission
```

#### Building the Docker Image

Build the Docker image with:

```bash
docker build -t mystreamlitapp .
```

This command builds a Docker image named `mystreamlitapp` from the Dockerfile in your project directory.

#### Running the Streamlit App

Run the app using:

```bash
docker run -p 8501:8501 mystreamlitapp
```

This command starts a container from the `mystreamlitapp` image and maps port 8501 from the container to port 8501 on your host machine.

### Accessing the Application

Open a web browser and navigate to `http://localhost:8501`. You should now see the Streamlit app running.

## Contributing

If you're open to contributions, provide instructions for how others can contribute to your project. Otherwise, you can omit this section.

## License

Specify the license under which your project is released, or refer to the LICENSE file in the repository.

---

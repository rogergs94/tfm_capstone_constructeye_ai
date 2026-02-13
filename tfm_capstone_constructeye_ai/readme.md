# ConstructEye AI: Real-Time PPE Compliance Monitoring

**Advanced Safety Oversight through Computer Vision**

!["ConstructEye_AI"](/images/constructeye_black.jpg)

**Author:** Roger González
**Program:** MSc in Data Science, Big Data & Business Analytics
**Repository:** `tfm_capstone_computer_vision_ppe`
**Tech Stack:** Python, Ultralytics (YOLOv8n), PyTorch, FastAPI, Streamlit, AWS S3, AWS ECS

## Overview

This project presents an end-to-end Computer Vision solution designed to automate safety monitoring in construction environments. The system detects the following classes using a custom-trained **YOLOv8n** model for fast inference:

- Helmets
- No-helmets
- Vests
- No-vests
- Person

The architecture is built on a decoupled microservices-ready approach, ensuring scalability and efficient real-time processing of worker safety compliance.

---

## Cloud Deployment & Infrastructure

To ensure professional-grade reliability and scalability, the system is designed for cloud-native deployment on **Amazon Web Services (AWS)**:

- **AWS ECS (Elastic Container Service):** The backend (FastAPI) and frontend (Streamlit) are containerized using Docker and deployed via ECS. This allows for seamless scaling and management of the microservices.
- **AWS S3 (Simple Storage Service):** A dedicated S3 bucket is integrated to serve as the centralized data lake. It stores:
  - **Processed Video Evidence:** Automated uploads of safety violation clips.
  - **Inference Logs:** CSV files organized by session ID for historical audit and Big Data analysis.
- **Compute Optimization:** Inferences are optimized for CPU-based instances to balance performance and cost-efficiency.

[Image of a microservices architecture diagram showing the interaction between a client, an API gateway, a processing service, and cloud storage]

---

## System Architecture

The application is divided into three main components:

1.  **Inference Engine (Backend):** A **FastAPI** server that handles frame-by-frame processing, executes the YOLOv8 model, and applies business logic to verify PPE compliance through geometric intersection algorithms.
2.  **Live Dashboard (Frontend):** A **Streamlit** interface utilizing **WebRTC** for low-latency video streaming, providing users with real-time visual feedback and safety metrics. You can also upload videos and images to inference and do deep analysis by frame and classes.
3.  **Cloud Storage & Logging:** Integration with **AWS S3** for persistent storage of processed video evidence and **CSV logs**, organized by unique session IDs for traceability.

---

## Tech Stack

- **Language:** Python 3.10+
- **Deep Learning:** Ultralytics (YOLOv8), PyTorch (CPU optimized).
- **Computer Vision:** OpenCV (Headless).
- **Web Frameworks:** FastAPI, Streamlit, Streamlit-WebRTC.
- **Cloud Infrastructure:** AWS (S3, EC2).
- **Environment Management:** Conda (Dev) / Pip (Prod).

---

## 🚀 Installation & Setup

To run this web application, you will need to execute both the Backend (FastAPI) and the Frontend (Streamlit) services.

To launch these services you will need separate terminals:

- Frontend: go to the directory and run the app.py (frontend) in your terminal with this command: _streamlit run app.py_
- Backend: you can either upsload this code to a docker container or just run it in another terminal with this code: _uvicorn main:app --reload_

### 1. Clone the Repository

```bash
git clone [https://github.com/rogergs94/tfm_capstone_constructeye_ai.git](https://github.com/rogergs94/tfm_capstone_constructeye_ai.git)
cd tfm_capstone_constructeye_ai
```

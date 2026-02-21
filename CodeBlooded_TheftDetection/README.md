# 🛡️ Petpooja Smart POS Security Dashboard

**A real-time AI surveillance system for Point-of-Sale (POS) monitoring, built for Hack.X.**

This project utilizes advanced computer vision and pose estimation to monitor cashier stations. It dynamically tracks the state of the cash drawer, distinguishes between cashiers and customers using spatial zoning, and alerts management to suspicious activities—such as unauthorized hands entering an open cash register.

## ✨ Key Features

* **Real-Time Drawer State Detection:** Utilizes a custom-trained model via Roboflow Inference to instantly detect whether the POS cash drawer is `OPEN` or `CLOSED`.
* **Spatial Zoning:** Interactive UI to map out custom "Cashier" and "Customer" zones based on the specific camera angle of the store.
* **Role Identification:** Classifies individuals as Cashiers or Customers based on their physical presence within the defined spatial zones.
* **Intrusion Detection (Pose Estimation):** Leverages YOLOv8 pose estimation to track human keypoints (wrists). If a wrist is detected intersecting with an open cash drawer bounding box, the system triggers a "HAND IN DRAWER" alert.
* **Live Metrics Dashboard:** Built with Streamlit to provide an easy-to-read, real-time breakdown of current store activity and security status.

## 🛠️ Tech Stack

* **UI Framework:** [Streamlit](https://streamlit.io/)
* **Computer Vision:** [OpenCV](https://opencv.org/)
* **Pose Estimation:** [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (`yolov8n-pose.pt`)
* **Object Detection:** [Roboflow Inference API](https://roboflow.com/)
* **Language:** Python 3.x

# HackIA'26 Group 3

> **UMONS 8th Residential Workshop** - May 16–17, 2026 <br>
> *XAI and GenAI for SmartHomes & Cities*
>
> This project was built during a **2-day hackathon** organized by the University of Mons.


## Context

Michel is a firefighter who also cares for his father, who lives alone with Alzheimer's. This project builds a **SafeAI** system to assist him on two fronts:

- **In the field**: real-time wildfire detection and fire/smoke localization with bounding boxes, accompanied by visual explanations (Grad-CAM / XAI).
- **At home**: face recognition for access control, detection of commonly misplaced personal objects (keys, phone, bag…), and automatic fall detection with instant alerts.

## Authors

- Tristan CLEMMEN 
- Pauline DELMOTTE
- Leonardo FERRETTI
- Harrison NGUYEN
- Arnaud VANEUKEM 

---

## Repository Structure

```
HackIA26_Groupe3/
├── app/                                Flask web application
├── src/                                Training and evaluation scripts
├── configs/                            Training configuration files
├── requirements.txt
└── HackIA26_Presentation_Groupe3.pdf   Project presentation deck
```

---

## Installation

**Requirements:** Python 3.10+, CUDA 12.8 (wheels are compiled for cu128).

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
cd app
python app.py
```

On startup, models are pre-loaded and the face recognition login screen is shown.

### Adding authorized faces

Drop one JPG/PNG photo per person into `app/authorized_faces/`. The filename becomes the displayed name (`jean_dupont.jpg` → *Jean Dupont*).

### Video sources

Each module (fire, objects, fall) accepts either **webcam** (select in the UI) or an **uploaded video file**.


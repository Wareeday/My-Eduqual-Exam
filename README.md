🧠 Neural Signal Acquisition & Preprocessing Engine

Production-ready EEG acquisition and preprocessing pipeline built in Python using BrainFlow.
Designed for real-time neural data streaming, signal conditioning, and feature extraction — suitable for research, BCI systems, and ML pipelines.

📌 Overview

This project provides a modular architecture for:

Real-time EEG data streaming

Signal preprocessing (bandpass filtering)

Feature extraction

Clean session lifecycle management

Synthetic board testing for development without hardware

The system is structured to be scalable, maintainable, and ready for integration into larger AI or BCI systems.

🏗 System Architecture
NeuralSignalAcquisition
│
├── Board Initialization
│   ├── BoardShim
│   └── BrainFlowInputParams
│
├── Streaming Layer
│   ├── prepare_session()
│   ├── start_stream()
│   ├── collect_data()
│   ├── stop_stream()
│   └── release_session()
│
├── Preprocessing Layer
│   ├── Bandpass Filtering
│   └── Channel-wise Processing
│
└── Feature Extraction Layer
    ├── Mean
    ├── Standard Deviation
    ├── RMS
    └── Signal Range

🚀 Key Features

Real-time EEG acquisition

Synthetic board support for development

Configurable bandpass filtering

Channel-wise signal processing

Statistical feature extraction

Structured logging

Clean shutdown handling

Modular OOP design

🛠 Tech Stack

Python 3.9+

BrainFlow

NumPy

Logging

📦 Installation
1. Clone Repository
git clone https://github.com/your-username/neural-signal-acquisition.git
cd neural-signal-acquisition

2. Create Virtual Environment
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies
pip install brainflow numpy

▶️ Usage

Run the main script:

python "Neural acquisition.PY"


Example output:

INFO:__main__:Started EEG streaming
Acquired data shape: (16, 250)
INFO:__main__:Stopped EEG streaming

Extracted Features:
{
  "mean": ...,
  "std": ...,
  "range": ...,
  "rms": ...
}

⚙️ Configuration

By default, the system uses:

BoardIds.SYNTHETIC_BOARD


This enables development without physical EEG hardware.

To connect to a real device, modify the board ID:

board_id = BoardIds.YOUR_DEVICE


Ensure proper hardware configuration in BrainFlowInputParams.

🧪 Signal Processing Details
Bandpass Filter

Low Cut Frequency: 1 Hz

High Cut Frequency: 40 Hz

Filter Type: Butterworth

Applied per EEG channel

Purpose:

Remove DC drift

Eliminate high-frequency noise

Preserve cognitive frequency bands

📊 Feature Extraction

For each EEG channel, the system computes:

Mean

Standard Deviation

Root Mean Square (RMS)

Signal Range

These features are suitable for:

Brain-Computer Interface systems

Cognitive workload detection

Emotion recognition

ML model training

Neurofeedback systems

🧠 Production Considerations

Proper session lifecycle management prevents memory leaks

Designed to integrate with ML pipelines

Easily extendable for:

Power spectral density analysis

Frequency band extraction (Alpha, Beta, Gamma)

Artifact removal (ICA)

Real-time dashboards

Cloud deployment

⚠️ Notes

You may see a BrainFlow cleanup warning during interpreter shutdown:

Exception ignored in: <function BoardShim.__del__>


This occurs during Python shutdown and does not affect functionality.

🔒 Scalability & Extension

Future roadmap:

Real-time visualization dashboard

REST API wrapper

Kafka-based streaming

ML inference integration

Containerized deployment (Docker)

Kubernetes-based scaling

📁 Project Structure
.
├── Neural acquisition.PY
├── README.md
└── .venv/
